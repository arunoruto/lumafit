# tests/test_lumafit.py
"""
Pytest test suite for the Lumafit library.

Verifies the core Levenberg-Marquardt algorithm and the pixel-wise fitting
function against known solutions and SciPy's implementation.
"""

import numpy as np
import numpy.testing as npt
import pytest
from lumafit import _EPS, levenberg_marquardt_core, levenberg_marquardt_pixelwise
from numba import jit
from scipy.optimize import least_squares

# --- Test Models (Numba JIT-able) ---
# These models now define the function signature func(p, t, *args_additional)
# where `t` is always the explicit independent variable array.


@jit(nopython=True, cache=True)
def model_exponential_decay(p, t, *args_additional):
    """
    Exponential decay model: y = p[0]*exp(-t/p[1]) + p[2]*exp(-t/p[3])

    Parameters p: [Amplitude1, DecayConstant1, Amplitude2, DecayConstant2]
    """
    # args_additional not used by this model, but must be accepted.
    term1 = np.zeros_like(t, dtype=np.float64)
    # Check for non-zero p[1] before division. np.abs(scalar) is scalar, so direct if is fine.
    if np.abs(p[1]) > 1e-12:
        term1 = p[0] * np.exp(-t / p[1])
    term2 = np.zeros_like(t, dtype=np.float64)
    # Check for non-zero p[3] before division.
    if np.abs(p[3]) > 1e-12:
        term2 = p[2] * np.exp(-t / p[3])
    return term1 + term2


@jit(nopython=True, cache=True)
def model_polarization(p, t, *args_additional):
    """
    Example polarization model: y = p[0] * sin(t_rad)^p[1] * cos(t_rad/2)^p[2] * sin(t_rad - p[3]_rad)

    Parameters p: [Amplitude, Exponent1, Exponent2, PhaseInDegrees]
    """
    # args_additional not used by this model, but must be accepted.
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
# These functions now accept (p, t, *args_additional)


@jit(nopython=True, cache=True)
def analytic_jacobian_exp_decay(p, t, *args_additional):
    """
    Analytical Jacobian for model_exponential_decay(p, t, *args_additional).
    J[i, j] = d(model_exponential_decay(p, t)) / d(p[j])
    """
    # args_additional not used by this Jacobian, but must be accepted.
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


@jit(nopython=True, cache=True)
def analytic_jacobian_polarization(p, t, *args_additional):
    """
    Analytical Jacobian for model_polarization(p, t, *args_additional).
    J[i, j] = d(model_polarization(p, t)) / d(p[j])
    """
    # args_additional not used by this Jacobian, but must be accepted.
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
# We need to capture the `lumafit_args_additional` for our models.
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
    # model_func now needs to be called with its own `t_scipy` and `*lumafit_args_additional`
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
        "t": t_data,  # Still keep t_data in fixture for test setup, as it's passed explicitly to LM functions
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
        "t": t_data,  # Still keep t_data in fixture for test setup
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_polarization,
        "jac_analytic": analytic_jacobian_polarization,
    }


# --- Test Configs ---
LMBA_TOL_CONFIG = {"tol_g": 1e-7, "tol_p": 1e-7, "tol_c": 1e-7, "max_iter": 1000}
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

    lmba_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # Lumafit `args` will be an empty tuple for `*args_additional`
    lumafit_args_additional = ()

    # Run the LMba fit on clean (no-noise) data using finite difference Jacobian
    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        d["model"],
        d["t"],  # Pass t explicitly
        p_initial,
        target_y=d["y_clean"],  # Pass y_clean as target_y
        weights=None,
        jac_func=None,
        args=lumafit_args_additional,  # Pass args_additional
        **lmba_params,
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

    lmba_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # Lumafit `args` will be an empty tuple for `*args_additional`
    lumafit_args_additional = ()

    # LMba fit (using default Finite Difference Jacobian)
    p_fit_lmba, _, chi2_lmba, iters_lmba, conv_lmba = levenberg_marquardt_core(
        d["model"],
        d["t"],  # Pass t explicitly
        p_initial,
        target_y=y_noisy,  # Pass y_noisy as target_y
        weights=weights_arr,
        jac_func=None,
        args=lumafit_args_additional,  # Pass args_additional
        **lmba_params,
    )
    assert conv_lmba, (
        f"LMba with weights failed (iters={iters_lmba}, chi2={chi2_lmba:.2e})"
    )

    # Scipy fit (using its default Finite Difference Jacobian)
    # The `args` tuple for SciPy's `least_squares` fun/jac now includes `t_scipy` and `lumafit_args_additional`
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
    t_data_common = d["t"]  # Common independent variable data
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
            # Model now takes `p, t, *args_additional`. `args_additional` is empty tuple here.
            y_clean_pixel = d["model"](p_pixel_true, t_data_common, ())
            noise_pixel = rng.normal(0, 0.01, size=depth).astype(np.float64)
            y_data_3d[r_idx, c_idx, :] = (y_clean_pixel + noise_pixel).astype(
                np.float64
            )

    p0_global = (d["p_true"] * 0.9).astype(np.float64)

    lmba_pixel_params = LMBA_TOL_CONFIG.copy()
    lmba_pixel_params.update(
        {"max_iter": 300, "tol_g": 1e-6, "tol_p": 1e-6, "tol_c": 1e-6}
    )

    # Lumafit `args_for_each_pixel` will be an empty tuple
    lumafit_args_for_each_pixel_empty = ()

    # Test pixelwise with default Finite Difference Jacobian
    p_res, cov_res, chi2_res, n_iter_res, conv_res = levenberg_marquardt_pixelwise(
        d["model"],
        t_data_common,  # Pass t_common explicitly
        p0_global,
        target_y_3d=y_data_3d,
        jac_func=None,
        args_for_each_pixel=lumafit_args_for_each_pixel_empty,  # Pass empty tuple
        **lmba_pixel_params,
    )

    assert conv_res.shape == (rows, cols)
    assert p_res.shape == (rows, cols, d["p_true"].shape[0])
    assert cov_res.shape == (rows, cols, d["p_true"].shape[0], d["p_true"].shape[0])
    assert chi2_res.shape == (rows, cols)
    assert n_iter_res.shape == (rows, cols)

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

    @jit(nopython=True, cache=True)
    def model_constant(p, t, *args_additional):
        return np.full_like(t, p[0], dtype=np.float64)

    t_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_data = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    p_initial = np.array([1.0], dtype=np.float64)
    p_true = np.array([5.0], dtype=np.float64)

    lmba_params_strict = LMBA_TOL_CONFIG.copy()
    lmba_params_strict.update(
        {"tol_g": 1e-9, "tol_p": 1e-9, "tol_c": 1e-9, "max_iter": 50}
    )

    # Lumafit `args` will be an empty tuple for `*args_additional`
    lumafit_args_additional = ()

    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        model_constant,
        t_data,  # Pass t explicitly
        p_initial,
        target_y=y_data,
        jac_func=None,
        args=lumafit_args_additional,  # Pass args_additional
        **lmba_params_strict,
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

    lmba_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # Lumafit `args` will be an empty tuple for `*args_additional`
    lumafit_args_additional = ()

    # LMba fit with weights (using default Finite Difference Jacobian)
    p_fit_lmba_w, _, chi2_lmba_w, iters_lmba_w, conv_lmba_w = levenberg_marquardt_core(
        d["model"],
        d["t"],  # Pass t explicitly
        p_initial_test,
        target_y=y_noisy,
        weights=weights_arr,
        jac_func=None,
        args=lumafit_args_additional,  # Pass args_additional
        **lmba_params,
    )
    assert conv_lmba_w, (
        f"LMba with weights failed (iters={iters_lmba_w}, chi2={chi2_lmba_w})"
    )

    # Scipy fit with weights (using its default Finite Difference Jacobian)
    # The `args` tuple for SciPy's `least_squares` fun/jac now includes `t_scipy` and `lumafit_args_additional`
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

    # LMba fit NO weights (using default Finite Difference Jacobian)
    p_fit_lmba_nw, _, chi2_lmba_nw, iters_lmba_nw, conv_lmba_nw = (
        levenberg_marquardt_core(
            d["model"],
            d["t"],  # Pass t explicitly
            p_initial_test,
            target_y=y_noisy,
            weights=None,
            jac_func=None,
            args=lumafit_args_additional,  # Pass args_additional
            **LMBA_TOL_CONFIG,
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
    # Lumafit `args` will be an empty tuple for `*args_additional`
    lumafit_args_additional = ()

    # --- Run 1: LMba with Analytical Jacobian ---
    lmba_params_analytic = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params_analytic["dp_ratio"] = 1e-7

    (
        p_fit_lmba_analytic,
        _,
        chi2_lmba_analytic,
        iters_lmba_analytic,
        conv_lmba_analytic,
    ) = levenberg_marquardt_core(
        d["model"],
        d["t"],  # Pass t explicitly
        p_initial,
        target_y=y_noisy,
        weights=weights_arr,
        jac_func=jac_analytic_func,
        args=lumafit_args_additional,  # Pass args_additional
        **lmba_params_analytic,
    )
    assert conv_lmba_analytic, (
        f"LMba (Analytic) failed: iter={iters_lmba_analytic}, chi2={chi2_lmba_analytic:.2e}"
    )

    # --- Run 2: LMba with Finite Difference Jacobian ---
    lmba_params_fd = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params_fd["dp_ratio"] = 1e-7

    p_fit_lmba_fd, _, chi2_lmba_fd, iters_lmba_fd, conv_lmba_fd = (
        levenberg_marquardt_core(
            d["model"],
            d["t"],  # Pass t explicitly
            p_initial,
            target_y=y_noisy,
            weights=weights_arr,
            jac_func=None,
            args=lumafit_args_additional,  # Pass args_additional
            **lmba_params_fd,
        )
    )
    assert conv_lmba_fd, (
        f"LMba (FD) failed: iter={iters_lmba_fd}, chi2={chi2_lmba_fd:.2e}"
    )

    # --- Run 3: Scipy with Analytical Jacobian ---
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

    # --- Compare the results from the three runs ---
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
