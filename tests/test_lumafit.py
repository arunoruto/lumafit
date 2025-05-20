import numpy as np
import numpy.testing as npt
import pytest

# Assuming lumafit module is installed or in PYTHONPATH
from lumafit import _EPS, levenberg_marquardt_core, levenberg_marquardt_pixelwise
from numba import jit
from scipy.optimize import least_squares  # Import for comparison


# --- Test Models (Numba JIT-able) ---
@jit(nopython=True, cache=True)
def model_exponential_decay(t, p):
    """y = p[0]*exp(-t/p[1]) + p[2]*exp(-t/p[3])"""
    term1 = np.zeros_like(t, dtype=np.float64)
    if np.abs(p[1]) > 1e-12:
        term1 = p[0] * np.exp(-t / p[1])
    term2 = np.zeros_like(t, dtype=np.float64)
    if np.abs(p[3]) > 1e-12:
        term2 = p[2] * np.exp(-t / p[3])
    return term1 + term2


@jit(nopython=True, cache=True)
def model_polarization(t, p):
    """y = p[0] * sin(t_rad)^p[1] * cos(t_rad/2)^p[2] * sin(t_rad - p3_rad)"""
    t_rad = t * np.pi / 180.0
    p3_rad = p[3] * np.pi / 180.0
    sin_t_rad_arr = np.sin(t_rad)
    term1_base = sin_t_rad_arr.copy()
    mask_problematic_base = (np.abs(term1_base) < _EPS) & (p[1] < 0.0)
    term1_base[mask_problematic_base] = _EPS
    term1 = np.power(term1_base, p[1])
    term2 = np.power(np.cos(t_rad / 2.0), p[2])
    term3 = np.sin(t_rad - p3_rad)
    return p[0] * term1 * term2 * term3


# --- Analytical Jacobian Functions (Numba JIT-able) ---
# Based on derivatives from previous discussion


@jit(nopython=True, cache=True)
def analytic_jacobian_exp_decay(t, p):
    """
    Analytical Jacobian for model_exponential_decay(t, p).
    J[i, j] = d(model_exponential_decay(t[i], p)) / d(p[j])
    """
    m = t.shape[0]
    n = p.shape[0]  # Should be 4
    J = np.empty((m, n), dtype=t.dtype)

    # Derivative terms (handle potential division by zero)
    exp_t_p1 = np.zeros_like(t)
    if np.abs(p[1]) > 1e-12:
        inv_p1 = 1.0 / p[1]
        exp_t_p1 = np.exp(-t * inv_p1)
        inv_p1_sq = inv_p1**2

        # d/dp[0]: exp(-t/p[1])
        J[:, 0] = exp_t_p1
        # d/dp[1]: p[0] * exp(-t/p[1]) * (t/p[1]**2)
        J[:, 1] = p[0] * exp_t_p1 * (t * inv_p1_sq)
    else:
        J[:, 0] = 0.0
        J[:, 1] = 0.0

    exp_t_p3 = np.zeros_like(t)
    if np.abs(p[3]) > 1e-12:
        inv_p3 = 1.0 / p[3]
        exp_t_p3 = np.exp(-t * inv_p3)
        inv_p3_sq = inv_p3**2

        # d/dp[2]: exp(-t/p[3])
        J[:, 2] = exp_t_p3
        # d/dp[3]: p[2] * exp(-t/p[3]) * (t/p[3]**2)
        J[:, 3] = p[2] * exp_t_p3 * (t * inv_p3_sq)
    else:
        J[:, 2] = 0.0
        J[:, 3] = 0.0

    return J


@jit(nopython=True, cache=True)
def analytic_jacobian_polarization(t, p):
    """
    Analytical Jacobian for model_polarization(t, p).
    J[i, j] = d(model_polarization(t[i], p)) / d(p[j])
    """
    m = t.shape[0]
    n = p.shape[0]  # Should be 4
    J = np.empty((m, n), dtype=t.dtype)

    t_rad = t * np.pi / 180.0
    p3_rad = p[3] * np.pi / 180.0

    sin_t_rad = np.sin(t_rad)
    cos_t_rad_half = np.cos(t_rad / 2.0)
    sin_t_rad_minus_p3_rad = np.sin(t_rad - p3_rad)
    cos_t_rad_minus_p3_rad = np.cos(t_rad - p3_rad)

    # Base terms from the model (with guards for derivatives involving log)
    # Use a smaller epsilon for bases potentially going into log
    log_guard_eps = 1e-15  # Smaller guard for log arguments

    term_f_base_for_log = sin_t_rad.copy()
    mask_prob_f_log = (
        np.abs(term_f_base_for_log) < log_guard_eps
    )  # sin(t_rad) near zero
    term_f_base_for_log[mask_prob_f_log] = (
        log_guard_eps  # Replace with tiny number for log stability
    )

    # Compute terms using guarded base where needed for derivatives
    # term_f_for_log = np.power(term_f_base_for_log, p[1]) # Use guarded base for log derivative term

    term_g = np.power(cos_t_rad_half, p[2])
    term_h = sin_t_rad_minus_p3_rad

    # For product terms, compute the sin(t_rad)^p[1] using original base, but handle 0^pos -> 0
    term_f_for_product = (
        np.power(sin_t_rad, p[1]) if np.abs(p[1]) > _EPS else np.zeros_like(t)
    )  # Simplified 0^pos guard

    # d/dp[0] (p[0]): This is the product of the other three terms
    J[:, 0] = term_f_for_product * term_g * term_h

    # d/dp[1] (p[1]): Derivative involves ln(sin(t_rad))
    # Use the guarded base for the log calculation itself
    log_sin_t_rad = np.log(term_f_base_for_log)

    # Derivative: p[0] * [sin(t_rad)^p[1] * ln(sin(t_rad))] * term_g * term_h
    # Ensure term_f_for_product is used here, which handles 0^pos correctly
    J[:, 1] = p[0] * term_f_for_product * log_sin_t_rad * term_g * term_h

    # d/dp[2] (p[2]): Derivative involves ln(cos(t_rad/2))
    log_cos_t_rad_half = np.log(cos_t_rad_half)
    J[:, 2] = p[0] * term_f_for_product * term_g * log_cos_t_rad_half * term_h

    # d/dp[3] (p[3], in degrees): Derivative involves cos(t_rad - p3_rad) * (-pi/180)
    J[:, 3] = (
        p[0] * term_f_for_product * term_g * (-np.pi / 180.0 * cos_t_rad_minus_p3_rad)
    )

    # --- Numba-compatible way to replace NaNs/Infs ---
    # Iterate and check each element, or use np.where
    # J[~np.isfinite(J)] = 0.0 # <<-- This caused the NumbaTypeError

    # Use np.where: result = np.where(condition, value_if_true, value_if_false)
    J = np.where(np.isfinite(J), J, 0.0)  # Replace non-finite elements with 0.0

    return J


# --- Scipy Residual Function Wrapper ---
# Moved to top level
def residuals_for_scipy(p_scipy, model_func, t_scipy, y_scipy, weights_scipy_sqrt):
    residuals = model_func(t_scipy, p_scipy) - y_scipy
    if weights_scipy_sqrt is not None:
        return weights_scipy_sqrt * residuals
    return residuals


# --- Scipy Analytical Jacobian Wrapper ---
def scipy_analytic_jacobian_wrapper(
    p_scipy, model_func, t_scipy, y_scipy, weights_scipy_sqrt, jac_func
):  # Keep this signature, but pass fewer args
    # jac_func is OUR analytical Jacobian function, expects (t, p) and returns m x n J
    # It will be the LAST argument received from Scipy's args/kwargs
    J_analytic = jac_func(t_scipy, p_scipy)  # Use the passed jac_func here

    # The Jacobian required by Scipy is w.r.t the residual function.
    # If residual = sqrt(W) * (model - data), J_residual = sqrt(W) * J_model.
    if weights_scipy_sqrt is not None:
        # Apply weights column-wise to the Jacobian from our analytical function
        return weights_scipy_sqrt[:, np.newaxis] * J_analytic

    # If no weights, residual = model - data, so J_residual = J_model.
    return J_analytic


# --- Test Fixtures ---
@pytest.fixture(name="exp_decay_data_dict")
def exp_decay_data_fixture():
    p_true = np.array([5.0, 2.0, 2.0, 10.0], dtype=np.float64)
    t_data = np.linspace(0.1, 25, 100, dtype=np.float64)
    y_clean = model_exponential_decay(t_data, p_true)
    return {
        "t": t_data,
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_exponential_decay,
        "jac_analytic": analytic_jacobian_exp_decay,
    }


@pytest.fixture(name="polarization_data_dict")
def polarization_data_fixture():
    p_true = np.array([2.0, 3.0, 5.0, 18.0], dtype=np.float64)
    t_data = np.linspace(1.0, 89.0, 100, dtype=np.float64)
    y_clean = model_polarization(t_data, p_true)
    return {
        "t": t_data,
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_polarization,
        "jac_analytic": analytic_jacobian_polarization,
    }


# --- Test Configs ---
LMBA_TOL_CONFIG = {"tol_g": 1e-7, "tol_p": 1e-7, "tol_c": 1e-7, "max_iter": 1000}
SCIPY_TOL_CONFIG = {"ftol": 1e-7, "xtol": 1e-7, "gtol": 1e-7, "max_nfev": 2000}


# --- Test Functions ---


@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_no_noise(data_fixture_name, request):
    """Test models with no noise, expects precise parameter recovery."""
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result

    p_initial = (d["p_true"] * 0.9).astype(np.float64)
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    lmba_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # Explicitly test with finite difference Jacobian for the no-noise case
    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        d["model"],
        d["t"],
        d["y_clean"],
        p_initial,
        weights=None,
        jac_func=None,
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
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result  # Unpack dict

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

    # LMba fit (using default Finite Difference Jacobian)
    p_fit_lmba, _, chi2_lmba, iters_lmba, conv_lmba = levenberg_marquardt_core(
        d["model"],
        d["t"],
        y_noisy,
        p_initial,
        weights=weights_arr,
        jac_func=None,
        **lmba_params,
    )
    assert conv_lmba, (
        f"{d['model'].__name__} (noise) LMba failed: iter={iters_lmba}, chi2={chi2_lmba:.2e}"
    )

    # Scipy fit (using its default Finite Difference Jacobian)
    scipy_res = least_squares(
        residuals_for_scipy,
        p_initial,  # residuals_for_scipy is top-level now
        args=(d["model"], d["t"], y_noisy, sqrt_weights_arr),
        method="lm",  # Scipy's default 'lm' uses finite difference jacobian by default if jac=None
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res.success, (
        f"{d['model'].__name__} (noise) Scipy failed. Status: {scipy_res.status}, Msg: {scipy_res.message}"
    )
    p_fit_scipy = scipy_res.x
    chi2_scipy = np.sum(scipy_res.fun**2)

    # Compare results (Finite Difference LMba vs Scipy)
    # Relax tolerances for comparison between LMba and Scipy on noisy data
    # Chi2 check is commented out as requested
    # npt.assert_allclose(chi2_lmba, chi2_scipy, rtol=0.2,
    #                     err_msg=f"{d['model'].__name__} (noise) chi2 mismatch")

    # Parameters should still be reasonably close if both converged
    npt.assert_allclose(
        p_fit_lmba,
        p_fit_scipy,
        rtol=0.2,
        atol=0.1,
        err_msg=f"{d['model'].__name__} (noise) param mismatch",
    )


def test_pixelwise_fitting(exp_decay_data_dict):
    d = exp_decay_data_dict  # Unpack dict
    rows, cols, depth = 2, 2, d["t"].shape[0]  # Small 2x2 grid for testing prange

    y_data_3d = np.empty((rows, cols, depth), dtype=np.float64)
    p_true_pixels = np.empty((rows, cols, d["p_true"].shape[0]), dtype=np.float64)
    rng = np.random.default_rng(42)

    for r_idx in range(rows):
        for c_idx in range(cols):
            p_pixel_true = d["p_true"] * (
                1 + rng.uniform(-0.05, 0.05, size=d["p_true"].shape)
            )
            p_true_pixels[r_idx, c_idx, :] = p_pixel_true.astype(np.float64)
            y_clean_pixel = d["model"](d["t"], p_pixel_true)
            noise_pixel = rng.normal(0, 0.01, size=depth).astype(
                np.float64
            )  # Low noise
            y_data_3d[r_idx, c_idx, :] = (y_clean_pixel + noise_pixel).astype(
                np.float64
            )

    p0_global = (d["p_true"] * 0.9).astype(np.float64)

    lmba_pixel_params = LMBA_TOL_CONFIG.copy()
    lmba_pixel_params.update(
        {"max_iter": 300, "tol_g": 1e-6, "tol_p": 1e-6, "tol_c": 1e-6}
    )

    # Test pixelwise with default Finite Difference Jacobian
    p_res, cov_res, chi2_res, n_iter_res, conv_res = levenberg_marquardt_pixelwise(
        d["model"], d["t"], y_data_3d, p0_global, jac_func=None, **lmba_pixel_params
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
    def model_constant(t, p):
        return np.full_like(t, p[0], dtype=np.float64)

    t_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_data = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    p_initial = np.array([1.0], dtype=np.float64)
    p_true = np.array([5.0], dtype=np.float64)

    lmba_params_strict = LMBA_TOL_CONFIG.copy()
    lmba_params_strict.update(
        {"tol_g": 1e-9, "tol_p": 1e-9, "tol_c": 1e-9, "max_iter": 50}
    )

    # Test with default Finite Difference Jacobian
    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        model_constant, t_data, y_data, p_initial, jac_func=None, **lmba_params_strict
    )
    assert conv, (
        f"Singular Jacobian test failed to converge. Iter: {iters}, Chi2: {chi2:.2e}"
    )
    npt.assert_allclose(
        p_fit, p_true, atol=1e-8, err_msg="Singular Jacobian test param mismatch"
    )
    assert chi2 < 1e-15, f"Singular Jacobian test Chi2 too high: {chi2:.2e}"


def test_weights_effect_vs_scipy(exp_decay_data_dict):
    d = exp_decay_data_dict  # Unpack dict
    rng = np.random.default_rng(5)
    p_initial_test = (d["p_true"] * 0.8).astype(np.float64)

    noise_std_profile = np.ones_like(d["y_clean"], dtype=np.float64) * 0.1
    noise_std_profile[: len(d["y_clean"]) // 2] = 0.5
    noise = (rng.normal(0, 1.0, size=d["y_clean"].shape) * noise_std_profile).astype(
        np.float64
    )
    y_noisy = (d["y_clean"] + noise).astype(np.float64)

    weights_arr = (1.0 / (noise_std_profile**2 + _EPS)).astype(np.float64)
    sqrt_weights_arr = np.sqrt(weights_arr)

    # LMba fit with weights (using default Finite Difference Jacobian)
    p_fit_lmba_w, _, chi2_lmba_w, iters_lmba_w, conv_lmba_w = levenberg_marquardt_core(
        d["model"],
        d["t"],
        y_noisy,
        p_initial_test,
        weights=weights_arr,
        jac_func=None,
        **LMBA_TOL_CONFIG,
    )
    assert conv_lmba_w, (
        f"LMba with weights failed (iters={iters_lmba_w}, chi2={chi2_lmba_w})"
    )

    # Scipy fit with weights (using its default Finite Difference Jacobian)
    scipy_res_w = least_squares(
        residuals_for_scipy,
        p_initial_test,  # residuals_for_scipy is top-level now
        args=(d["model"], d["t"], y_noisy, sqrt_weights_arr),
        method="lm",
        # jac=None # Scipy uses FD if jac=None
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res_w.success, (
        f"Scipy with weights failed. Status: {scipy_res_w.status}, Msg: {scipy_res_w.message}"
    )
    p_fit_scipy_w = scipy_res_w.x
    chi2_scipy_w = np.sum(scipy_res_w.fun**2)

    # Compare weighted fits (FD LMba vs FD Scipy)
    # Chi2 check is commented out as requested
    # npt.assert_allclose(chi2_lmba_w, chi2_scipy_w, rtol=0.2,
    #                     err_msg=f"Weighted chi2 mismatch. LMba: {chi2_lmba_w}, Scipy: {chi2_scipy_w}")

    npt.assert_allclose(
        p_fit_lmba_w,
        p_fit_scipy_w,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Weighted param mismatch. LMba: {p_fit_lmba_w}, Scipy: {p_fit_scipy_w}",
    )

    # LMba fit NO weights (using default Finite Difference Jacobian)
    p_fit_lmba_nw, _, chi2_lmba_nw, iters_lmba_nw, conv_lmba_nw = (
        levenberg_marquardt_core(
            d["model"],
            d["t"],
            y_noisy,
            p_initial_test,
            weights=None,
            jac_func=None,
            **LMBA_TOL_CONFIG,
        )
    )
    assert conv_lmba_nw, (
        f"LMba no weights failed (iters={iters_lmba_nw}, chi2={chi2_lmba_nw})"
    )

    # Check that weighted and unweighted fits (from LMba, both using FD) give different parameters
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
    and against Scipy's LM using the analytical Jacobian.
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

    jac_analytic_func = d[
        "jac_analytic"
    ]  # This is the analytical function from the fixture

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
        d["t"],
        y_noisy,
        p_initial,
        weights=weights_arr,
        jac_func=jac_analytic_func,  # Pass analytical Jacobian function
        **lmba_params_analytic,
    )
    assert conv_lmba_analytic, (
        f"{d['model'].__name__} (Analytic) LMba failed: iter={iters_lmba_analytic}, chi2={chi2_lmba_analytic:.2e}"
    )

    # --- Run 2: LMba with Finite Difference Jacobian ---
    lmba_params_fd = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params_fd["dp_ratio"] = 1e-7  # dp_ratio is used here

    p_fit_lmba_fd, _, chi2_lmba_fd, iters_lmba_fd, conv_lmba_fd = (
        levenberg_marquardt_core(
            d["model"],
            d["t"],
            y_noisy,
            p_initial,
            weights=weights_arr,
            jac_func=None,  # Use finite difference
            **lmba_params_fd,
        )
    )
    assert conv_lmba_fd, (
        f"{d['model'].__name__} (FD) LMba failed: iter={iters_lmba_fd}, chi2={chi2_lmba_fd:.2e}"
    )

    # --- Run 3: Scipy with Analytical Jacobian ---

    # Define the Scipy Jacobian wrapper *inside* the test function
    # This allows it to access jac_analytic_func from the outer scope
    def scipy_analytic_jacobian_wrapper_local(
        p_scipy, model_func, t_scipy, y_scipy, weights_scipy_sqrt
    ):
        # This wrapper is called by Scipy with p_scipy, and the items from the 'args' tuple
        # It needs to call the actual analytical jacobian function (jac_analytic_func)
        # which is available in the outer scope.

        # Call our *actual* analytical jacobian function
        J_analytic = jac_analytic_func(
            t_scipy, p_scipy
        )  # <<< Uses jac_analytic_func from outer scope

        # Apply weights as required by Scipy's API for the residual Jacobian
        if weights_scipy_sqrt is not None:
            return weights_scipy_sqrt[:, np.newaxis] * J_analytic
        return J_analytic

    # The args tuple for residuals_for_scipy. Does NOT include jac_analytic_func.
    scipy_fun_args = (d["model"], d["t"], y_noisy, sqrt_weights_arr)

    scipy_res_analytic = least_squares(
        residuals_for_scipy,
        p_initial,  # Use the standard residuals function
        args=scipy_fun_args,  # Pass args needed ONLY by residuals_for_scipy
        method="lm",
        jac=scipy_analytic_jacobian_wrapper_local,  # Tell Scipy to use our local wrapper
        # No jac_args needed!
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res_analytic.success, (
        f"{d['model'].__name__} (Analytic) Scipy failed. Status: {scipy_res_analytic.status}, Msg: {scipy_res_analytic.message}"
    )
    p_fit_scipy_analytic = scipy_res_analytic.x
    chi2_scipy_analytic = np.sum(scipy_res_analytic.fun**2)

    # --- Compare the results from the three runs ---
    npt.assert_allclose(
        p_fit_lmba_analytic,
        p_fit_lmba_fd,
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{d['model'].__name__}: LMba Analytic vs FD param mismatch",
    )
    # npt.assert_allclose(
    #     chi2_lmba_analytic,
    #     chi2_lmba_fd,
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg=f"{d['model'].__name__}: LMba Analytic vs FD chi2 mismatch",
    # )

    npt.assert_allclose(
        p_fit_lmba_analytic,
        p_fit_scipy_analytic,
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{d['model'].__name__}: LMba Analytic vs Scipy Analytic param mismatch",
    )
    # npt.assert_allclose(
    #     chi2_lmba_analytic,
    #     chi2_scipy_analytic,
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg=f"{d['model'].__name__}: LMba Analytic vs Scipy Analytic chi2 mismatch",
    # )

    npt.assert_allclose(
        p_fit_lmba_fd,
        p_fit_scipy_analytic,
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{d['model'].__name__}: LMba FD vs Scipy Analytic param mismatch",
    )
    # npt.assert_allclose(
    #     chi2_lmba_fd,
    #     chi2_scipy_analytic,
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg=f"{d['model'].__name__}: LMba FD vs Scipy Analytic chi2 mismatch",
    # )
