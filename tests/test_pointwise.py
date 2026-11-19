"""Tests for the pointwise solver on the CPU and, where available, on Vulkan."""

import math

import numpy as np
import numpy.testing as npt
import pytest
from numba import njit

from lumafit import levenberg_marquardt_pixelwise, make_pixelwise_fitter

P_TRUE = np.array([5.0, 2.0, 2.0, 10.0])
P0 = np.array([4.0, 1.5, 1.5, 8.0])
T = np.linspace(0.1, 25, 100)
WEIGHTS = np.linspace(1.0, 3.0, T.size)


@njit
def decay(p, t):
    return p[0] * math.exp(-t / p[1]) + p[2] * math.exp(-t / p[3])


@njit
def decay_array(p, *args):
    t = args[0]
    return p[0] * np.exp(-t / p[1]) + p[2] * np.exp(-t / p[3])


@pytest.fixture(scope="module")
def cube():
    """An 8 x 8 cube of noisy decays with one masked pixel."""
    rng = np.random.default_rng(0)
    p = P_TRUE * rng.uniform(0.9, 1.1, (8, 8, 4))
    y = p[..., 0:1] * np.exp(-T / p[..., 1:2]) + p[..., 2:3] * np.exp(-T / p[..., 3:4])
    y += rng.normal(0, 0.02, y.shape)
    y[0, 1, 5] = np.nan
    return y


@pytest.fixture(scope="module")
def reference(cube):
    """The array solver's result for the same weighted fit."""
    return levenberg_marquardt_pixelwise(
        decay_array, P0, cube, weights_1d=WEIGHTS, args_for_each_pixel=(T,)
    )


def vulkan_device():
    """A Vulkan device to test on, or skip; a real GPU is preferred."""
    nv = pytest.importorskip("numba_vulkan", exc_type=ImportError)
    devices = nv.list_devices()
    if not devices:
        pytest.skip("no Vulkan device")
    return devices[0]


def check(result, reference, rtol_p, rtol_cov):
    p, cov, chi2, n_iter, converged = result
    p_ref, cov_ref, chi2_ref, _, converged_ref = reference
    fitted = converged_ref

    assert p.shape == p_ref.shape and cov.shape == cov_ref.shape
    npt.assert_allclose(p[fitted], p_ref[fitted], rtol=rtol_p)
    npt.assert_allclose(cov[fitted], cov_ref[fitted], rtol=rtol_cov, atol=1e-8)
    npt.assert_allclose(chi2[fitted], chi2_ref[fitted], rtol=rtol_p)
    # The masked pixel is skipped, everything else converges.
    assert np.isnan(p[0, 1]).all() and np.isnan(cov[0, 1]).all()
    assert np.isnan(chi2[0, 1]) and n_iter[0, 1] == 0
    npt.assert_array_equal(converged, fitted)


@pytest.mark.parametrize(
    ("dtype", "rtol_p", "rtol_cov"),
    [(np.float64, 1e-5, 1e-3), (np.float32, 5e-3, 5e-2)],
)
def test_cpu_matches_array_solver(cube, reference, dtype, rtol_p, rtol_cov):
    fit = make_pixelwise_fitter(decay, 4, dtype)
    result = fit(P0, cube, T, weights_1d=WEIGHTS)
    assert result[0].dtype == dtype
    check(result, reference, rtol_p, rtol_cov)


@pytest.mark.parametrize(
    ("dtype", "rtol_p", "rtol_cov"),
    [(np.float64, 1e-5, 1e-3), (np.float32, 5e-3, 5e-2)],
)
def test_vulkan_matches_array_solver(cube, reference, dtype, rtol_p, rtol_cov):
    device = vulkan_device()
    if dtype == np.float64 and not device.float64:
        pytest.skip(f"{device.name} has no float64")
    fit = make_pixelwise_fitter(decay, 4, dtype, target="vulkan")
    result = fit(P0, cube, T, weights_1d=WEIGHTS, device=device.index)
    assert result[0].dtype == dtype
    check(result, reference, rtol_p, rtol_cov)


def test_rejects_bad_arguments(cube):
    with pytest.raises(ValueError, match="target"):
        make_pixelwise_fitter(decay, 4, target="cuda")
    with pytest.raises(ValueError, match="dtype"):
        make_pixelwise_fitter(decay, 4, dtype=np.int32)
    fit = make_pixelwise_fitter(decay, 4)
    with pytest.raises(ValueError, match="p0"):
        fit(P0[:3], cube, T)
    with pytest.raises(ValueError, match="one per sample"):
        fit(P0, cube, T[:-1])
