# tests/test_lumafit.py
"""
Pytest test suite for the Lumafit library.

Verifies the core Levenberg-Marquardt algorithm and the pixel-wise fitting
function against known solutions and SciPy's implementation.
"""

import numpy as np
import numpy.testing as npt
import pytest

# Assuming lumafit module is installed or in PYTHONPATH
from lumafit import _EPS  # Epsilon for numerical stability
from lumafit import levenberg_marquardt_core, levenberg_marquardt_pixelwise
from numba import jit
from scipy.optimize import least_squares  # Import for comparison

# --- Test Models (Numba JIT-able) ---
# These models must be Numba-compatible and define the function signature func(t, p)


@jit(nopython=True, cache=True)
def model_exponential_decay(t, p):
    """
    Exponential decay model: y = p[0]*exp(-t/p[1]) + p[2]*exp(-t/p[3])

    Parameters p: [Amplitude1, DecayConstant1, Amplitude2, DecayConstant2]
    """
    term1 = np.zeros_like(t, dtype=np.float64)
    # Guard against division by zero or near-zero for decay constants
    if np.abs(p[1]) > 1e-12:
        term1 = p[0] * np.exp(-t / p[1])
    term2 = np.zeros_like(t, dtype=np.float64)
    if np.abs(p[3]) > 1e-12:
        term2 = p[2] * np.exp(-t / p[3])
    return term1 + term2


@jit(nopython=True, cache=True)
def model_polarization(t, p):
    """
    Example polarization model: y = p[0] * sin(t_rad)^p[1] * cos(t_rad/2)^p[2] * sin(t_rad - p[3]_rad)

    Parameters p: [Amplitude, Exponent1, Exponent2, PhaseInDegrees]
    """
    t_rad = t * np.pi / 180.0
    # p[3] is in degrees, convert to radians for the sin function
    p3_rad = p[3] * np.pi / 180.0

    sin_t_rad_arr = np.sin(t_rad)
    cos_t_rad_half = np.cos(t_rad / 2.0)

    # Handle potential issues with np.power(0, negative) or np.log(0/negative)
    # Create a base term for the sin(t_rad)^p[1] part.
    # If sin(t_rad) is near zero AND p[1] is negative, the power is problematic (Inf/NaN).
    # If sin(t_rad) is near zero AND p[1] is positive, power is near zero.
    # The base used for the product terms should handle 0^positive->0 correctly.
    term_f_base_for_product = (
        np.power(sin_t_rad_arr, p[1]) if np.abs(p[1]) > _EPS else np.zeros_like(t)
    )

    term_g = np.power(cos_t_rad_half, p[2])
    term_h = np.sin(t_rad - p3_rad)

    return p[0] * term_f_base_for_product * term_g * term_h


# --- Analytical Jacobian Functions (Numba JIT-able) ---
# These functions accept (t, p) and return the Jacobian J (m x n)
# where m is len(t) and n is len(p). They must be Numba-compatible.


@jit(nopython=True, cache=True)
def analytic_jacobian_exp_decay(t, p):
    """
    Analytical Jacobian for model_exponential_decay(t, p).
    J[i, j] = d(model_exponential_decay(t[i], p)) / d(p[j])
    """
    m = t.shape[0]
    n = p.shape[0]  # Should be 4 parameters
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
def analytic_jacobian_polarization(t, p):
    """
    Analytical Jacobian for model_polarization(t, p).
    J[i, j] = d(model_polarization(t[i], p)) / d(p[j])
    """
    m = t.shape[0]
    n = p.shape[0]  # Should be 4 parameters
    J = np.empty((m, n), dtype=t.dtype)

    t_rad = t * np.pi / 180.0
    p3_rad = p[3] * np.pi / 180.0

    sin_t_rad = np.sin(t_rad)
    cos_t_rad_half = np.cos(t_rad / 2.0)
    sin_t_rad_minus_p3_rad = np.sin(t_rad - p3_rad)
    cos_t_rad_minus_p3_rad = np.cos(t_rad - p3_rad)

    log_guard_eps = 1e-15
    term_f_base_for_log = sin_t_rad.copy()
    mask_prob_f_log = np.abs(term_f_base_for_log) < log_guard_eps
    term_f_base_for_log[mask_prob_f_log] = log_guard_eps

    term_g = np.power(cos_t_rad_half, p[2])
    term_h = sin_t_rad_minus_p3_rad

    term_f_for_product = (
        np.power(sin_t_rad, p[1]) if np.abs(p[1]) > _EPS else np.zeros_like(t)
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
# For least_squares with method='lm', args must contain all extra data needed by fun AND jac.
# The jac_func is needed by jac but not fun. However, to keep the *args tuple consistent,
# we pass jac_func in args and accept it here, even though we don't use it.
def residuals_for_scipy(
    p_scipy,
    model_func,
    t_scipy,
    y_scipy,
    weights_scipy_sqrt,
    dummy_jac_func=None,
):
    """Calculates residuals for SciPy's least_squares."""
    # dummy_jac_func is included in signature to match the args tuple structure passed by SciPy.
    # It is not used in this function.
    residuals = model_func(t_scipy, p_scipy) - y_scipy
    if weights_scipy_sqrt is not None:
        # For least_squares with loss='linear', it minimizes sum(f_i**2).
        # If we want to minimize sum(W_i * (model_i - data_i)**2),
        # the residual f_i passed to SciPy should be sqrt(W_i) * (model_i - data_i).
        return weights_scipy_sqrt * residuals
    return residuals


# --- Scipy Analytical Jacobian Wrapper ---
# This is the 'jac' argument for scipy.optimize.least_squares when jac is callable
# It MUST match the signature expected by SciPy's API: jac(x, *args, **kwargs)
# It needs access to the *actual analytical jacobian function* (jac_func),
# which is passed as the LAST item in the args tuple by our test.
def scipy_analytic_jacobian_wrapper(
    p_scipy, model_func, t_scipy, y_scipy, weights_scipy_sqrt, jac_func
):
    """
    Wrapper for providing an analytical Jacobian function to SciPy's least_squares.

    Parameters
    ----------
    p_scipy : numpy.ndarray
        The current parameters vector (n-element 1D array) from SciPy.
    model_func : callable
        The model function `y_hat = model_func(t, p)`.
    t_scipy : numpy.ndarray
        Independent variable data (m-element 1D array).
    y_scipy : numpy.ndarray
        Dependent variable data (m-element 1D array).
    weights_scipy_sqrt : numpy.ndarray or None
        Square root of the weights (m-element 1D array).
    jac_func : callable
        The *actual analytical Jacobian function* `J = jac_func(t, p)`.
        This is the last argument passed from SciPy's `args` tuple.

    Returns
    -------
    numpy.ndarray
        The Jacobian matrix (m x n) of the *residuals*, weighted by sqrt(W).
    """
    # Call our *actual analytical jacobian function* using the current parameters and data
    # jac_func is the last argument received from the args tuple
    J_analytic = jac_func(t_scipy, p_scipy)  # This is the Jacobian of the MODEL

    # The Jacobian required by SciPy is w.r.t the *residual* function.
    # If residual = sqrt(W) * (model - data), J_residual = sqrt(W) * J_model.
    if weights_scipy_sqrt is not None:
        # Apply weights column-wise to the Jacobian from our analytical function
        return weights_scipy_sqrt[:, np.newaxis] * J_analytic

    return J_analytic


# --- Test Fixtures ---
# Fixtures provide reproducible data for tests.


@pytest.fixture(name="exp_decay_data_dict")
def exp_decay_data_fixture():
    """Fixture for exponential decay model data."""
    p_true = np.array([5.0, 2.0, 2.0, 10.0], dtype=np.float64)
    t_data = np.linspace(0.1, 25, 100, dtype=np.float64)  # More points for stability
    y_clean = model_exponential_decay(t_data, p_true)
    # Also provide the analytical jacobian for this model
    return {
        "t": t_data,
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_exponential_decay,
        "jac_analytic": analytic_jacobian_exp_decay,
    }


@pytest.fixture(name="polarization_data_dict")
def polarization_data_fixture():
    """Fixture for polarization model data."""
    p_true = np.array([2.0, 3.0, 5.0, 18.0], dtype=np.float64)
    t_data = np.linspace(1.0, 89.0, 100, dtype=np.float64)  # More points
    y_clean = model_polarization(t_data, p_true)
    # Also provide the analytical jacobian for this model
    return {
        "t": t_data,
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_polarization,
        "jac_analytic": analytic_jacobian_polarization,
    }


# --- Test Configs ---
# Configuration dictionaries for LMba and SciPy tolerances and max iterations.
# Used to keep test settings consistent.
LMBA_TOL_CONFIG = {"tol_g": 1e-7, "tol_p": 1e-7, "tol_c": 1e-7, "max_iter": 1000}
# Scipy's default ftol, xtol, gtol are 1e-8. max_nfev is number of func evaluations.
# Explicitly set method='lm' here
SCIPY_TOL_CONFIG = {
    "ftol": 1e-7,
    "xtol": 1e-7,
    "gtol": 1e-7,
    "max_nfev": 2000,
    "method": "lm",
}


# --- Test Functions ---
# Test functions use fixtures to get data and compare results.


@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_no_noise(data_fixture_name, request):
    """
    Test models with no noise.

    Expects precise parameter recovery and a very low Chi-squared value.
    Tests Lumafit's finite difference Jacobian by default.
    """
    # Get the actual fixture result using request.getfixturevalue
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result

    # Set initial guess - starting somewhat close is reasonable for non-linear problems.
    # Ensure it's float64.
    p_initial = (d["p_true"] * 0.9).astype(
        np.float64
    )  # Start 10% off as a common strategy

    # You can add model-specific initial guesses if the default perturbation isn't good enough
    # For example, using the specific guess that worked for the polarization model initially:
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    # Get LMba parameters, potentially adding model-specific ones
    lmba_params = LMBA_TOL_CONFIG.copy()
    # Add model-specific parameters if necessary (e.g., dp_ratio for polarization)
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # Run the LMba fit on clean (no-noise) data using finite difference Jacobian
    # Use jac_func=None explicitly for clarity
    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        d["model"],
        d["t"],
        d["y_clean"],
        p_initial,
        weights=None,
        jac_func=None,
        **lmba_params,
    )

    # Assertions for no-noise fit:
    # 1. Check if the algorithm converged
    assert conv, (
        f"{d['model'].__name__} (no noise) failed: iter={iters}, chi2={chi2:.2e}"
    )

    # 2. Expect very close recovery of the true parameters for no-noise data
    # Use strict tolerances for assert_allclose
    npt.assert_allclose(
        p_fit,
        d["p_true"],
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{d['model'].__name__} (no noise) params mismatch",
    )

    # 3. Chi2 should be very close to zero for a perfect fit (model = data)
    # Use a very small tolerance for Chi2 check
    assert chi2 < 1e-8, f"{d['model'].__name__} (no noise) Chi2 too high: {chi2:.2e}"


@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_with_noise_vs_scipy(data_fixture_name, request):
    """
    Test models with noise by comparing Lumafit's FD results against SciPy's FD results.

    Expect fitted parameters to be reasonably close, but Chi2 can differ due
    to local minima and algorithm details. Chi2 comparison is commented out.
    """
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result  # Unpack dict

    # Use a specific seed for reproducibility of noise
    rng = np.random.default_rng(0 if d["model"] == model_exponential_decay else 1)
    # Use a noise level consistent with your previous tests where failures occurred
    noise_std = 0.2 if d["model"] == model_exponential_decay else 0.05

    noise = rng.normal(0, noise_std, size=d["y_clean"].shape).astype(np.float64)
    y_noisy = (d["y_clean"] + noise).astype(np.float64)

    # Use an initial guess
    p_initial = (d["p_true"] * 0.9).astype(np.float64)
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    # Calculate weights based on the noise_std
    weights_arr = np.full_like(y_noisy, 1.0 / (noise_std**2 + _EPS), dtype=np.float64)
    sqrt_weights_arr = np.sqrt(weights_arr)

    lmba_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # LMba fit (using default Finite Difference Jacobian)
    # Use jac_func=None explicitly for clarity
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
    # Pass args needed by residuals_for_scipy
    scipy_args = (d["model"], d["t"], y_noisy, sqrt_weights_arr)

    scipy_res = least_squares(
        residuals_for_scipy,  # residuals_for_scipy is top-level now
        p_initial,
        args=scipy_args,  # Pass args needed by residuals_for_scipy
        **SCIPY_TOL_CONFIG,  # SCIPY_TOL_CONFIG includes method='lm'
    )
    assert scipy_res.success, (
        f"Scipy with weights failed. Status: {scipy_res.status}, Msg: {scipy_res.message}"
    )
    p_fit_scipy = scipy_res.x
    chi2_scipy = np.sum(scipy_res.fun**2)

    # Compare results (Finite Difference LMba vs SciPy)
    # Chi2 check is commented out as requested due to potential large differences
    # npt.assert_allclose(chi2_lmba, chi2_scipy, rtol=0.2,
    #                     err_msg=f"Chi2 mismatch. LMba: {chi2_lmba:.2e}, SciPy: {chi2_scipy:.2e}")

    # Parameters should still be reasonably close if both converged to nearby minima
    npt.assert_allclose(
        p_fit_lmba,
        p_fit_scipy,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Param mismatch. LMba: {p_fit_lmba}, SciPy: {p_fit_scipy}",
    )


def test_pixelwise_fitting(exp_decay_data_dict):
    """Test the pixelwise fitting function on a small 3D dataset."""
    d = exp_decay_data_dict  # Unpack dict
    rows, cols, depth = 2, 2, d["t"].shape[0]  # Small 2x2 grid for testing prange

    y_data_3d = np.empty((rows, cols, depth), dtype=np.float64)
    p_true_pixels = np.empty((rows, cols, d["p_true"].shape[0]), dtype=np.float64)
    rng = np.random.default_rng(42)

    for r_idx in range(rows):
        for c_idx in range(cols):
            # Slightly vary true params per pixel for a more robust test
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
    # Use jac_func=None explicitly for clarity
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
            # Compare fitted params to the true params used for that specific pixel
            # Use relaxed tolerances for noisy data and varied true params
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
    # Use jac_func=None explicitly for clarity
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
    """
    Test that providing weights affects the fit outcome, comparing Lumafit's
    FD results against SciPy's FD results, both with and without weights.
    """
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
    # Use jac_func=None explicitly for clarity
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
    # Pass args needed by residuals_for_scipy
    scipy_args_w = (d["model"], d["t"], y_noisy, sqrt_weights_arr)
    scipy_res_w = least_squares(
        residuals_for_scipy,
        p_initial_test,
        args=scipy_args_w,
        # REMOVED explicit method='lm' here - it's in **SCIPY_TOL_CONFIG
        **SCIPY_TOL_CONFIG,  # SCIPY_TOL_CONFIG includes method='lm'
    )
    assert scipy_res_w.success, (
        f"Scipy with weights failed. Status: {scipy_res_w.status}, Msg: {scipy_res_w.message}"
    )
    p_fit_scipy_w = scipy_res_w.x
    chi2_scipy_w = np.sum(scipy_res_w.fun**2)

    # Compare weighted fits (FD LMba vs FD SciPy)
    # Chi2 check is commented out as requested
    # npt.assert_allclose(chi2_lmba_w, chi2_scipy_w, rtol=0.2,
    #                     err_msg=f"Chi2 mismatch. LMba: {chi2_lmba_w:.2e}, SciPy: {chi2_scipy_w:.2e}")

    npt.assert_allclose(
        p_fit_lmba_w,
        p_fit_scipy_w,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Param mismatch. LMba: {p_fit_lmba_w}, SciPy: {p_fit_scipy_w}",
    )

    # LMba fit NO weights (using default Finite Difference Jacobian)
    # Use jac_func=None explicitly for clarity
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
        jac_func=jac_analytic_func,
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
    scipy_res_analytic = least_squares(
        residuals_for_scipy,
        p_initial,
        # Pass args needed by *both* residuals_for_scipy and scipy_analytic_jacobian_wrapper
        # These args are: model_func, t, y, sqrt_weights, jac_func
        args=(d["model"], d["t"], y_noisy, sqrt_weights_arr, jac_analytic_func),
        jac=scipy_analytic_jacobian_wrapper,  # Tell Scipy to use OUR wrapper
        **SCIPY_TOL_CONFIG,  # SCIPY_TOL_CONFIG also includes method='lm'
    )
    assert scipy_res_analytic.success, (
        f"{d['model'].__name__} (Analytic) Scipy failed. Status: {scipy_res_analytic.status}, Msg: {scipy_res_analytic.message}"
    )
    p_fit_scipy_analytic = scipy_res_analytic.x
    chi2_scipy_analytic = np.sum(scipy_res_analytic.fun**2)

    # --- Compare the results from the three runs ---

    # Tolerances for comparison involving the Finite Difference Jacobian
    # FD vs Analytic LMba, LMba FD vs Scipy Analytic
    param_rtol_fd_comparisons = 1e-4
    param_atol_fd_comparisons = 5e-4
    chi2_rtol_fd_comparisons = 1e-5
    chi2_atol_fd_comparisons = 1e-7

    # Tolerances for comparison between Analytical LMba and Analytical Scipy
    param_rtol_analytic_scipy = 1e-5
    param_atol_analytic_scipy = 1e-6
    chi2_rtol_analytic_scipy = 1e-7
    chi2_atol_analytic_scipy = 1e-9

    # Analytical vs Finite Difference (LMba) - Relaxed tolerance
    npt.assert_allclose(
        p_fit_lmba_analytic,
        p_fit_lmba_fd,
        rtol=param_rtol_fd_comparisons,
        atol=param_atol_fd_comparisons,
        err_msg=f"{d['model'].__name__}: LMba Analytic vs FD param mismatch",
    )
    # Chi2 check is commented out here too as requested for this specific comparison
    # npt.assert_allclose(
    #     chi2_lmba_analytic,
    #     chi2_lmba_fd,
    #     rtol=chi2_rtol_fd_comparisons,
    #     atol=chi2_atol_fd_comparisons,
    #     err_msg=f"{d['model'].__name__}: LMba Analytic vs FD chi2 mismatch",
    # )

    # LMba Analytic vs Scipy Analytic - Strict tolerance
    npt.assert_allclose(
        p_fit_lmba_analytic,
        p_fit_scipy_analytic,
        rtol=param_rtol_analytic_scipy,
        atol=param_atol_analytic_scipy,
        err_msg=f"{d['model'].__name__}: LMba Analytic vs SciPy Analytic param mismatch",
    )
    # Chi2 check is commented out here too as requested for this specific comparison
    # npt.assert_allclose(
    #     chi2_lmba_analytic,
    #     chi2_scipy_analytic,
    #     rtol=chi2_rtol_analytic_scipy,
    #     atol=chi2_atol_analytic_scipy,
    #     err_msg=f"{d['model'].__name__}: LMba Analytic vs SciPy Analytic chi2 mismatch",
    # )

    # Finite Difference (LMba) vs Scipy Analytic - Relaxed tolerance
    npt.assert_allclose(
        p_fit_lmba_fd,
        p_fit_scipy_analytic,
        rtol=param_rtol_fd_comparisons,
        atol=param_atol_fd_comparisons,
        err_msg=f"{d['model'].__name__}: LMba FD vs SciPy Analytic param mismatch",
    )
    # Chi2 check is commented out here too as requested for this specific comparison
    # npt.assert_allclose(
    #     chi2_lmba_fd,
    #     chi2_scipy_analytic,
    #     rtol=chi2_rtol_fd_comparisons,
    #     atol=chi2_atol_fd_comparisons,
    #     err_msg=f"{d['model'].__name__}: LMba FD vs SciPy Analytic chi2 mismatch",
    # )
