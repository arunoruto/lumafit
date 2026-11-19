# tests/test_lumafit.py
"""
Pytest test suite for the Lumafit library.

Verifies the core Levenberg-Marquardt algorithm and the pixel-wise fitting
function against known solutions and SciPy's implementation.
"""

import numpy as np
import numpy.testing as npt
import pytest
from lumafit import _EPS, FitResult, fit_cube, fit_curve
from numba import jit
from scipy.optimize import least_squares

# --- Test Models (Numba JIT-able) ---
# These models now define the function signature func(p, *args)
# The `args` tuple will typically contain `t` as its first element for these models.


@jit(nopython=True, cache=True, fastmath=True)  # Added fastmath
def model_exponential_decay(p, *args):
    """
    Exponential decay model: y = p[0]*exp(-t/p[1]) + p[2]*exp(-t/p[3])

    Parameters p: [Amplitude1, DecayConstant1, Amplitude2, DecayConstant2]
    """
    # Assuming `t` is the first element in `args`
    t: np.typing.NDArray[np.float64] = args[0]  # Explicitly hint type for Numba

    term1 = np.zeros_like(t, dtype=np.float64)
    if (
        np.abs(p[1]) > 1e-12
    ):  # p[1] is scalar, so np.abs(scalar) is scalar. This comparison is fine.
        term1 = p[0] * np.exp(-t / p[1])
    term2 = np.zeros_like(t, dtype=np.float64)
    if np.abs(p[3]) > 1e-12:  # p[3] is scalar. This comparison is fine.
        term2 = p[2] * np.exp(-t / p[3])
    return term1 + term2


@jit(nopython=True, cache=True, fastmath=True)  # Added fastmath
def model_polarization(p, *args):
    """
    Example polarization model: y = p[0] * sin(t_rad)^p[1] * cos(t_rad/2)^p[2] * sin(t_rad - p[3]_rad)

    Parameters p: [Amplitude, Exponent1, Exponent2, PhaseInDegrees]
    """
    # Assuming `t` is the first element in `args`
    t: np.typing.NDArray[np.float64] = args[0]  # Explicitly hint type for Numba

    t_rad = t * np.pi / 180.0
    p3_rad = p[3] * np.pi / 180.0

    sin_t_rad_arr = np.sin(t_rad)
    cos_t_rad_half = np.cos(t_rad / 2.0)

    # Use np.where for conditional assignment based on an array condition
    term_f_base_for_product = np.where(
        np.abs(sin_t_rad_arr) > _EPS,  # condition is element-wise array comparison
        np.power(
            sin_t_rad_arr, p[1]
        ),  # p[1] is scalar, so power works element-wise on array
        np.zeros_like(t),  # value if false
    )

    term_g = np.power(
        cos_t_rad_half, p[2]
    )  # p[2] is scalar, so power works element-wise on array
    term_h = np.sin(t_rad - p3_rad)

    return p[0] * term_f_base_for_product * term_g * term_h


# --- Analytical Jacobian Functions (Numba JIT-able) ---
# These functions accept (p, *args) and return the Jacobian J (m x n)


@jit(nopython=True, cache=True, fastmath=True)  # Added fastmath
def analytic_jacobian_exp_decay(p, *args):
    """
    Analytical Jacobian for model_exponential_decay(p, t, *args_additional).
    J[i, j] = d(model_exponential_decay(p, t)) / d(p[j])
    """
    # Assuming `t` is the first element in `args`
    t: np.typing.NDArray[np.float64] = args[0]  # Explicitly hint type for Numba

    m = t.shape[0]
    n = p.shape[0]
    J = np.empty((m, n), dtype=t.dtype)

    # Derivative terms (handle potential division by zero or near-zero for decay constants)

    # d/dp[0]: exp(-t/p[1])
    # d/dp[1]: p[0] * exp(-t/p[1]) * (t/p[1]**2)
    if np.abs(p[1]) > 1e-12:
        inv_p1 = 1.0 / p[1]
        exp_t_p1 = np.exp(-t * inv_p1)
        J[:, 0] = exp_t_p1
        inv_p1_sq = inv_p1**2
        J[:, 1] = p[0] * exp_t_p1 * (t * inv_p1_sq)
    else:
        J[:, 0] = 0.0
        J[:, 1] = 0.0

    # d/dp[2]: exp(-t/p[3])
    # d/dp[3]: p[2] * exp(-t/p[3]) * (t/p[3]**2)
    if np.abs(p[3]) > 1e-12:
        inv_p3 = 1.0 / p[3]
        exp_t_p3 = np.exp(-t * inv_p3)
        J[:, 2] = exp_t_p3
        inv_p3_sq = inv_p3**2
        J[:, 3] = p[2] * exp_t_p3 * (t * inv_p3_sq)
    else:
        J[:, 2] = 0.0
        J[:, 3] = 0.0

    # Replace any non-finite results (NaN/Inf) with 0.0 for numerical stability in LM
    # Use np.where which is Numba-compatible for this assignment
    J = np.where(np.isfinite(J), J, 0.0)

    return J


@jit(nopython=True, cache=True, fastmath=True)  # Added fastmath
def analytic_jacobian_polarization(p, *args):
    """
    Analytical Jacobian for model_polarization(p, t, *args_additional).
    J[i, j] = d(model_polarization(p, t)) / d(p[j])
    """
    # Assuming `t` is the first element in `args`
    t: np.typing.NDArray[np.float64] = args[0]  # Explicitly hint type for Numba

    m = t.shape[0]
    n = p.shape[0]
    J = np.empty((m, n), dtype=t.dtype)

    t_rad = t * np.pi / 180.0
    p3_rad = p[3] * np.pi / 180.0

    sin_t_rad = np.sin(t_rad)
    cos_t_rad_half = np.cos(t_rad / 2.0)
    sin_t_rad_minus_p3_rad = np.sin(t_rad - p3_rad)
    cos_t_rad_minus_p3_rad = np.cos(t_rad - p3_rad)

    log_guard_eps = 1e-15
    term_f_base_for_log = np.where(  # Using np.where here for array condition
        np.abs(sin_t_rad) < log_guard_eps, log_guard_eps, sin_t_rad
    )

    term_g = np.power(cos_t_rad_half, p[2])
    term_h = np.sin(t_rad - p3_rad)

    term_f_for_product = np.where(  # Using np.where here for array condition
        np.abs(sin_t_rad) > _EPS, np.power(sin_t_rad, p[1]), np.zeros_like(t)
    )

    J[:, 0] = term_f_for_product * term_g * term_h

    log_sin_t_rad = np.log(term_f_base_for_log)
    J[:, 1] = p[0] * term_f_for_product * log_sin_t_rad * term_g * term_h

    log_cos_t_rad_half = np.log(cos_t_rad_half)
    J[:, 2] = p[0] * term_f_for_product * term_g * log_cos_t_rad_half * term_h

    J[:, 3] = (
        p[0] * term_f_for_product * term_g * (-np.pi / 180.0 * cos_t_rad_minus_p3_rad)
    )

    J = np.where(np.isfinite(J), J, 0.0)

    return J


# --- Scipy Residual Function Wrapper ---
# This is the 'fun' argument for scipy.optimize.least_squares
# It MUST match the signature expected by SciPy's API: fun(x, *args, **kwargs)
# `target_y_scipy` corresponds to the data to fit against.
# `t_scipy` is passed explicitly here for these tests, as are `lumafit_args_additional`.
def residuals_for_scipy(
    p_scipy,
    model_func,
    target_y_scipy,
    weights_scipy_sqrt,
    dummy_jac_func,
    t_scipy,
    lumafit_args_additional,
):
    """Calculates residuals for SciPy's least_squares."""
    # model_func now needs to be called with `p`, `t_scipy`, and `*lumafit_args_additional`
    residuals = model_func(p_scipy, t_scipy, *lumafit_args_additional) - target_y_scipy
    if weights_scipy_sqrt is not None:
        return weights_scipy_sqrt * residuals
    return residuals


# --- Scipy Analytical Jacobian Wrapper ---
# This is the 'jac' argument for scipy.optimize.least_squares when jac is callable
# It MUST match the signature expected by SciPy's API: jac(x, *args, **kwargs)
def scipy_analytic_jacobian_wrapper(
    p_scipy,
    model_func,
    target_y_scipy,
    weights_scipy_sqrt,
    jac_func,
    t_scipy,
    lumafit_args_additional,
):
    """
    Wrapper for providing an analytical Jacobian function to SciPy's least_squares.
    """
    # Call our *actual analytical jacobian function* using the current parameters and data
    J_analytic = jac_func(p_scipy, t_scipy, *lumafit_args_additional)

    if weights_scipy_sqrt is not None:
        return weights_scipy_sqrt[:, np.newaxis] * J_analytic

    return J_analytic


# --- Test Fixtures ---


@pytest.fixture(name="exp_decay_data_dict")
def exp_decay_data_fixture():
    """Fixture for exponential decay model data."""
    p_true = np.array([5.0, 2.0, 2.0, 10.0], dtype=np.float64)
    t_data = np.linspace(0.1, 25, 100, dtype=np.float64)
    # Model now takes `p, t, *args_additional`. `args_additional` is empty tuple here.
    y_clean = model_exponential_decay(p_true, t_data, ())
    return {
        "t": t_data,  # Still keep t_data in fixture as it's passed explicitly to LM functions
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_exponential_decay,
        "jac_analytic": analytic_jacobian_exp_decay,
    }


@pytest.fixture(name="polarization_data_dict")
def polarization_data_fixture():
    """Fixture for polarization model data."""
    p_true = np.array([2.0, 3.0, 5.0, 18.0], dtype=np.float64)
    t_data = np.linspace(1.0, 89.0, 100, dtype=np.float64)
    # Model now takes `p, t, *args_additional`. `args_additional` is empty tuple here.
    y_clean = model_polarization(p_true, t_data, ())
    return {
        "t": t_data,  # Still keep t_data in fixture
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_polarization,
        "jac_analytic": analytic_jacobian_polarization,
    }


# --- Test Configs ---
LMBA_TOL_CONFIG = {
    "tol_g": 1e-7,
    "tol_p": 1e-7,
    "tol_c": 1e-7,
    "max_iter": 1000,
    "dp_ratio": 1e-8,
}
SCIPY_TOL_CONFIG = {
    "ftol": 1e-7,
    "xtol": 1e-7,
    "gtol": 1e-7,
    "max_nfev": 2000,
    "method": "lm",
}


# --- Test Functions ---


@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_no_noise(data_fixture_name, request):
    """
    Test models with no noise.
    Tests Lumafit's finite difference Jacobian by default.
    """
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result

    p_initial = (d["p_true"] * 0.9).astype(np.float64)
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    # Prepare specific LMba parameters for this test case
    core_lm_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        core_lm_params["dp_ratio"] = 1e-7  # Override dp_ratio if needed

    lumafit_args_additional = (d["t"],)  # `t_data` is now in `args` tuple

    p_fit, cov, chi2, iters, conv = fit_curve(
        model=d["model"],
        p0=p_initial,  # p0 is first explicit arg
        y=d["y_clean"],  # Pass y_clean as target_y
        weights=None,
        jac=None,
        args=lumafit_args_additional,  # Pass args_additional
        **core_lm_params,  # Pass parameters from the dictionary using **
    )

    assert conv, (
        f"{d['model'].__name__} (no noise) failed: iter={iters}, chi2={chi2:.2e}"
    )
    npt.assert_allclose(
        p_fit,
        d["p_true"],
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{d['model'].__name__} (no noise) params mismatch",
    )
    assert chi2 < 1e-8, f"{d['model'].__name__} (no noise) Chi2 too high: {chi2:.2e}"


@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_with_noise_vs_scipy(data_fixture_name, request):
    """
    Test models with noise by comparing Lumafit's FD results against SciPy's FD results.
    """
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result

    rng = np.random.default_rng(0 if d["model"] == model_exponential_decay else 1)
    noise_std = 0.2 if d["model"] == model_exponential_decay else 0.05

    noise = rng.normal(0, noise_std, size=d["y_clean"].shape).astype(np.float64)
    y_noisy = (d["y_clean"] + noise).astype(np.float64)

    p_initial = (d["p_true"] * 0.9).astype(np.float64)
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    weights_arr = np.full_like(y_noisy, 1.0 / (noise_std**2 + _EPS), dtype=np.float64)
    sqrt_weights_arr = np.sqrt(weights_arr)

    core_lm_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        core_lm_params["dp_ratio"] = 1e-7

    lumafit_args_additional = (d["t"],)  # `t_data` is now in `args` tuple for models

    p_fit_lmba, _, chi2_lmba, iters_lmba, conv_lmba = fit_curve(
        model=d["model"],
        p0=p_initial,  # p0 is first explicit arg
        y=y_noisy,  # Pass y_noisy as target_y
        weights=weights_arr,
        jac=None,
        args=lumafit_args_additional,  # Pass args_additional
        **core_lm_params,  # Pass parameters from the dictionary using **
    )
    assert conv_lmba, (
        f"LMba with weights failed (iters={iters_lmba}, chi2={chi2_lmba:.2e})"
    )

    scipy_args = (
        d["model"],
        y_noisy,
        sqrt_weights_arr,
        None,
        d["t"],
        lumafit_args_additional,
    )
    scipy_res = least_squares(
        residuals_for_scipy,
        p_initial,
        args=scipy_args,
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res.success, (
        f"Scipy with weights failed. Status: {scipy_res.status}, Msg: {scipy_res.message}"
    )
    p_fit_scipy = scipy_res.x
    chi2_scipy = np.sum(scipy_res.fun**2)

    npt.assert_allclose(
        p_fit_lmba,
        p_fit_scipy,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Param mismatch. LMba: {p_fit_lmba}, SciPy: {p_fit_scipy}",
    )


def test_pixelwise_fitting(exp_decay_data_dict):
    """Test the pixelwise fitting function on a small 3D dataset."""
    d = exp_decay_data_dict
    t_data_common = d["t"]
    rows, cols, depth = 2, 2, t_data_common.shape[0]

    y_data_3d = np.empty((rows, cols, depth), dtype=np.float64)
    p_true_pixels = np.empty((rows, cols, d["p_true"].shape[0]), dtype=np.float64)
    rng = np.random.default_rng(42)

    for r_idx in range(rows):
        for c_idx in range(cols):
            p_pixel_true = d["p_true"] * (
                1 + rng.uniform(-0.05, 0.05, size=d["p_true"].shape)
            )
            p_true_pixels[r_idx, c_idx, :] = p_pixel_true.astype(np.float64)
            y_clean_pixel = d["model"](p_pixel_true, t_data_common, ())
            noise_pixel = rng.normal(0, 0.01, size=depth).astype(np.float64)
            y_data_3d[r_idx, c_idx, :] = (y_clean_pixel + noise_pixel).astype(
                np.float64
            )

    p0_global = (d["p_true"] * 0.9).astype(np.float64)

    core_lm_pixel_params = LMBA_TOL_CONFIG.copy()
    core_lm_pixel_params.update(
        {"max_iter": 300, "tol_g": 1e-6, "tol_p": 1e-6, "tol_c": 1e-6}
    )

    lumafit_args_for_each_pixel_empty = (
        t_data_common,
    )  # Common `t` is always the first in this tuple

    p_res, cov_res, chi2_res, n_iter_res, conv_res = fit_cube(
        model=d["model"],
        p0=p0_global,  # p0 is first explicit arg
        cube=y_data_3d,
        weights=None,
        jac=None,
        args=lumafit_args_for_each_pixel_empty,
        **core_lm_pixel_params,  # Pass parameters from the dictionary using **
    )

    assert conv_res.shape == (rows, cols)
    assert p_res.shape == (rows, cols, d["p_true"].shape[0])
    assert cov_res.shape == (rows, cols, d["p_true"].shape[0], d["p_true"].shape[0])
    assert chi2_res.shape == (rows, cols)
    assert n_iter_res.shape == (rows, cols)  # Corrected NameError

    for r_idx in range(rows):
        for c_idx in range(cols):
            assert conv_res[r_idx, c_idx], (
                f"Pixel ({r_idx},{c_idx}) failed to converge."
            )
            npt.assert_allclose(
                p_res[r_idx, c_idx, :],
                p_true_pixels[r_idx, c_idx, :],
                rtol=0.2,
                atol=0.1,
                err_msg=f"Pixel ({r_idx},{c_idx}) params mismatch",
            )


def test_singular_jacobian_case():
    """Test behavior when Jacobian might lead to singular JtWJ initially (e.g., constant model)."""

    @jit(nopython=True, cache=True, fastmath=True)  # Added fastmath
    def model_constant(p, *args):
        # Assuming `t` is the first element in `args`
        t: np.typing.NDArray[np.float64] = args[0]  # Explicitly hint type for Numba
        return np.full_like(t, p[0], dtype=np.float64)

    t_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_data = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    p_initial = np.array([1.0], dtype=np.float64)
    p_true = np.array([5.0], dtype=np.float64)

    core_lm_params_strict = LMBA_TOL_CONFIG.copy()
    core_lm_params_strict.update(
        {"tol_g": 1e-9, "tol_p": 1e-9, "tol_c": 1e-9, "max_iter": 50}
    )

    lumafit_args_additional = (t_data,)  # `t_data` is now in `args` tuple

    p_fit, cov, chi2, iters, conv = fit_curve(
        model=model_constant,
        p0=p_initial,  # p0 is first explicit arg
        y=y_data,
        weights=None,
        jac=None,
        args=lumafit_args_additional,  # Pass args_additional
        **core_lm_params_strict,  # Pass parameters from the dictionary using **
    )
    assert conv, (
        f"Singular Jacobian test failed to converge. Iter: {iters}, Chi2: {chi2:.2e}"
    )
    npt.assert_allclose(
        p_fit, p_true, atol=1e-8, err_msg="Singular Jacobian test param mismatch"
    )
    assert chi2 < 1e-15, f"Singular Jacobian test Chi2 too high: {chi2:.2e}"


def test_weights_effect_vs_scipy(exp_decay_data_dict):
    """
    Test that providing weights affects the fit outcome, comparing Lumafit's
    FD results against SciPy's FD results, both with and without weights.
    """
    d = exp_decay_data_dict
    rng = np.random.default_rng(5)
    p_initial_test = (d["p_true"] * 0.8).astype(np.float64)

    noise_std_profile = np.ones_like(d["y_clean"], dtype=np.float64) * 0.1
    noise_std_profile[: len(d["y_clean"]) // 2] = 0.5
    noise = (rng.normal(0, 1.0, size=d["y_clean"].shape) * noise_std_profile).astype(
        np.float64
    )
    y_noisy = (d["y_clean"] + noise).astype(np.float64)

    weights_arr = np.full_like(
        y_noisy, 1.0 / (noise_std_profile**2 + _EPS), dtype=np.float64
    )
    sqrt_weights_arr = np.sqrt(weights_arr)

    core_lm_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        core_lm_params["dp_ratio"] = 1e-7

    lumafit_args_additional = (d["t"],)  # `t_data` is now in `args` tuple for models

    p_fit_lmba_w, _, chi2_lmba_w, iters_lmba_w, conv_lmba_w = fit_curve(
        model=d["model"],
        p0=p_initial_test,  # p0 is first explicit arg
        y=y_noisy,
        weights=weights_arr,
        jac=None,
        args=lumafit_args_additional,
        **core_lm_params,  # Pass parameters from the dictionary using **
    )
    assert conv_lmba_w, (
        f"LMba with weights failed (iters={iters_lmba_w}, chi2={chi2_lmba_w})"
    )

    scipy_args_w = (
        d["model"],
        y_noisy,
        sqrt_weights_arr,
        None,
        d["t"],
        lumafit_args_additional,
    )
    scipy_res_w = least_squares(
        residuals_for_scipy,
        p_initial_test,
        args=scipy_args_w,
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res_w.success, (
        f"Scipy with weights failed. Status: {scipy_res_w.status}, Msg: {scipy_res_w.message}"
    )
    p_fit_scipy_w = scipy_res_w.x
    chi2_scipy_w = np.sum(scipy_res_w.fun**2)

    npt.assert_allclose(
        p_fit_lmba_w,
        p_fit_scipy_w,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Param mismatch. LMba: {p_fit_lmba_w}, SciPy: {p_fit_scipy_w}",
    )

    p_fit_lmba_nw, _, chi2_lmba_nw, iters_lmba_nw, conv_lmba_nw = (
        fit_curve(
            model=d["model"],
            p0=p_initial_test,  # p0 is first explicit arg
            y=y_noisy,
            weights=None,
            jac=None,
            args=lumafit_args_additional,
            **core_lm_params,  # Pass parameters from the dictionary using **
        )
    )
    assert conv_lmba_nw, (
        f"LMba no weights failed (iters={iters_lmba_nw}, chi2={chi2_lmba_nw})"
    )

    diff_sum_params = np.sum(np.abs(p_fit_lmba_w - p_fit_lmba_nw))
    assert diff_sum_params > 1e-3, (
        "LMba: Weighted and unweighted parameters are too close, weights might not have a significant effect."
    )


@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_analytic_jacobian_vs_fd_vs_scipy(data_fixture_name, request):
    """
    Test fitting with analytical Jacobian vs finite difference Jacobian (LMba)
    and against SciPy's LM using the analytical Jacobian.
    Expect results to be very close.
    """
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result

    rng = np.random.default_rng(10 if d["model"] == model_exponential_decay else 11)
    noise_std = 0.01
    noise = rng.normal(0, noise_std, size=d["y_clean"].shape).astype(np.float64)
    y_noisy = (d["y_clean"] + noise).astype(np.float64)

    p_initial = (d["p_true"] * 0.9).astype(np.float64)
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    weights_arr = np.full_like(y_noisy, 1.0 / (noise_std**2 + _EPS), dtype=np.float64)
    sqrt_weights_arr = np.sqrt(weights_arr)

    jac_analytic_func = d["jac_analytic"]
    lumafit_args_additional = (d["t"],)  # `t_data` is now in `args` tuple for models

    core_lm_params_analytic = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        core_lm_params_analytic["dp_ratio"] = 1e-7

    (
        p_fit_lmba_analytic,
        _,
        chi2_lmba_analytic,
        iters_lmba_analytic,
        conv_lmba_analytic,
    ) = fit_curve(
        model=d["model"],
        p0=p_initial,  # p0 is first explicit arg
        y=y_noisy,
        weights=weights_arr,
        jac=jac_analytic_func,
        args=lumafit_args_additional,
        **core_lm_params_analytic,  # Pass parameters from the dictionary using **
    )
    assert conv_lmba_analytic, (
        f"LMba (Analytic) failed: iter={iters_lmba_analytic}, chi2={chi2_lmba_analytic:.2e}"
    )

    core_lm_params_fd = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        core_lm_params_fd["dp_ratio"] = 1e-7

    p_fit_lmba_fd, _, chi2_lmba_fd, iters_lmba_fd, conv_lmba_fd = (
        fit_curve(
            model=d["model"],
            p0=p_initial,  # p0 is first explicit arg
            y=y_noisy,
            weights=weights_arr,
            jac=None,
            args=lumafit_args_additional,
            **core_lm_params_fd,  # Pass parameters from the dictionary using **
        )
    )
    assert conv_lmba_fd, (
        f"LMba (FD) failed: iter={iters_lmba_fd}, chi2={chi2_lmba_fd:.2e}"
    )

    scipy_res_analytic = least_squares(
        residuals_for_scipy,
        p_initial,
        args=(
            d["model"],
            y_noisy,
            sqrt_weights_arr,
            jac_analytic_func,
            d["t"],
            lumafit_args_additional,
        ),
        jac=scipy_analytic_jacobian_wrapper,
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res_analytic.success, (
        f"Scipy (Analytic) failed. Status: {scipy_res_analytic.status}, Msg: {scipy_res_analytic.message}"
    )
    p_fit_scipy_analytic = scipy_res_analytic.x
    chi2_scipy_analytic = np.sum(scipy_res_analytic.fun**2)

    param_rtol_fd_comparisons = 1e-4
    param_atol_fd_comparisons = 5e-4
    param_rtol_analytic_scipy = 1e-5
    param_atol_analytic_scipy = 1e-6

    npt.assert_allclose(
        p_fit_lmba_analytic,
        p_fit_lmba_fd,
        rtol=param_rtol_fd_comparisons,
        atol=param_atol_fd_comparisons,
        err_msg=f"{d['model'].__name__}: LMba Analytic vs FD param mismatch",
    )

    npt.assert_allclose(
        p_fit_lmba_analytic,
        p_fit_scipy_analytic,
        rtol=param_rtol_analytic_scipy,
        atol=param_atol_analytic_scipy,
        err_msg=f"{d['model'].__name__}: LMba Analytic vs SciPy Analytic param mismatch",
    )

    npt.assert_allclose(
        p_fit_lmba_fd,
        p_fit_scipy_analytic,
        rtol=param_rtol_fd_comparisons,
        atol=param_atol_fd_comparisons,
        err_msg=f"{d['model'].__name__}: LMba FD vs SciPy Analytic param mismatch",
    )


# --- Regression tests ---


def _noisy_decay(d, noise_std=0.05, seed=3):
    rng = np.random.default_rng(seed)
    return d["y_clean"] + rng.normal(0, noise_std, d["y_clean"].shape)


def test_fd_covariance_matches_analytic(exp_decay_data_dict):
    """The finite difference Jacobian must be scaled like the analytic one."""
    d = exp_decay_data_dict
    y, p0, args = _noisy_decay(d), d["p_true"] * 0.9, (d["t"],)

    p_fd, cov_fd, *_ = fit_curve(d["model"], p0, y, args=args)
    p_an, cov_an, *_ = fit_curve(
        d["model"], p0, y, jac=d["jac_analytic"], args=args
    )

    npt.assert_allclose(p_fd, p_an, rtol=1e-6)
    npt.assert_allclose(cov_fd, cov_an, rtol=1e-4)


def test_weighted_fit_matches_scipy_tightly(exp_decay_data_dict):
    """Weights are 1/sigma**2, in the step and in the returned chi-squared."""
    d = exp_decay_data_dict
    y, p0, t = _noisy_decay(d), d["p_true"] * 0.9, d["t"]
    w = np.linspace(1.0, 100.0, y.shape[0])

    p_fit, _, chi2, _, conv = fit_curve(
        d["model"], p0, y, weights=w, args=(t,)
    )
    ref = least_squares(
        lambda p: np.sqrt(w) * (d["model"](p, t) - y), p0, method="lm", xtol=1e-12
    )

    assert conv
    npt.assert_allclose(p_fit, ref.x, rtol=1e-5)
    npt.assert_allclose(chi2, np.sum(ref.fun**2), rtol=1e-8)


def test_residual_function_without_target(exp_decay_data_dict):
    """With no target, the function output is minimized as residuals."""
    d = exp_decay_data_dict
    y, p0, t = _noisy_decay(d), d["p_true"] * 0.9, d["t"]

    @jit(nopython=True)
    def residuals(p, *args):
        return model_exponential_decay(p, args[0]) - args[1]

    p_res, *_ = fit_curve(residuals, p0, args=(t, y))
    p_fit, *_ = fit_curve(d["model"], p0, y, args=(t,))

    npt.assert_allclose(p_res, p_fit, rtol=1e-6)


def test_pixelwise_skips_nan_pixels(exp_decay_data_dict):
    """Masked pixels stay NaN/unconverged, the others are still fitted."""
    d = exp_decay_data_dict
    cube = np.tile(_noisy_decay(d), (2, 2, 1))
    cube[0, 1, 5] = np.nan

    p, cov, chi2, n_iter, conv = fit_cube(
        d["model"], d["p_true"] * 0.9, cube, args=(d["t"],)
    )

    assert np.isnan(p[0, 1]).all() and np.isnan(cov[0, 1]).all()
    assert np.isnan(chi2[0, 1]) and n_iter[0, 1] == 0 and not conv[0, 1]
    assert conv.sum() == 3
    npt.assert_allclose(p[1, 1], d["p_true"], rtol=0.1)
    npt.assert_array_equal(p[1, 1], p[0, 0])  # same curve, same fit


def test_converges_in_few_iterations(exp_decay_data_dict):
    """Guards the damping strategy: this fit takes MINPACK about 6 steps."""
    d = exp_decay_data_dict
    *_, n_iter, conv = fit_curve(
        d["model"], d["p_true"] * 0.9, _noisy_decay(d), args=(d["t"],)
    )
    assert conv and n_iter <= 10


# --- API ---


@jit(nopython=True)
def decay_plain(p, t):
    """A model with ordinary parameters instead of ``*args``."""
    return p[0] * np.exp(-t / p[1]) + p[2] * np.exp(-t / p[3])


def test_results_are_named_and_unpackable(exp_decay_data_dict):
    d = exp_decay_data_dict
    y, args = _noisy_decay(d), (d["t"],)

    result = fit_curve(decay_plain, d["p_true"] * 0.9, y, args=args)
    assert isinstance(result, FitResult)
    p, cov, chi2, n_iter, converged = result
    assert result.p is p and result.cov is cov
    assert (result.chi2, result.n_iter, result.converged) == (chi2, n_iter, converged)

    cube = fit_cube(decay_plain, d["p_true"] * 0.9, np.tile(y, (2, 2, 1)), args=args)
    assert isinstance(cube, FitResult)
    npt.assert_allclose(cube.p[1, 1], p)
    assert cube.converged.all() and cube.n_iter.shape == (2, 2)


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_accepts_other_input_dtypes(exp_decay_data_dict, dtype):
    """An integer or single precision guess is fitted in float64."""
    d = exp_decay_data_dict
    p0 = np.array([4, 2, 2, 9], dtype=dtype)
    y, args = _noisy_decay(d), (d["t"],)

    reference = fit_curve(decay_plain, p0.astype(np.float64), y, args=args)
    result = fit_curve(decay_plain, p0, y, args=args)
    assert result.p.dtype == np.float64 and result.converged
    npt.assert_allclose(result.p, reference.p, rtol=1e-12)

    cube = fit_cube(decay_plain, p0, np.tile(y, (2, 2, 1)).astype(np.float32), args=args)
    assert cube.p.dtype == np.float64 and cube.converged.all()
    npt.assert_allclose(cube.p[0, 0], reference.p, rtol=1e-4)
