import numpy as np
import numpy.testing as npt
import pytest
from lumafit import _EPS, levenberg_marquardt_core, levenberg_marquardt_pixelwise
from numba import jit
from scipy.optimize import least_squares


# --- Test Models (Numba JIT-able) ---
@jit(nopython=True, cache=True)
def model_exponential_decay(t, p):
    term1 = np.zeros_like(t, dtype=np.float64)
    # Add tolerance check for divisor
    if np.abs(p[1]) > 1e-12:
        term1 = p[0] * np.exp(-t / p[1])
    term2 = np.zeros_like(t, dtype=np.float64)
    # Add tolerance check for divisor
    if np.abs(p[3]) > 1e-12:
        term2 = p[2] * np.exp(-t / p[3])
    return term1 + term2


@jit(nopython=True, cache=True)
def model_polarization(t, p):
    t_rad = t * np.pi / 180.0
    p3_rad = p[3] * np.pi / 180.0
    sin_t_rad_arr = np.sin(t_rad)
    term1_base = sin_t_rad_arr.copy()
    # Handle 0**negative for np.power if p[1] can be negative
    mask_problematic_base = (np.abs(term1_base) < _EPS) & (p[1] < 0.0)
    term1_base[mask_problematic_base] = _EPS
    term1 = np.power(term1_base, p[1])
    term2 = np.power(np.cos(t_rad / 2.0), p[2])
    term3 = np.sin(t_rad - p3_rad)
    return p[0] * term1 * term2 * term3


# --- Scipy Residual Function Wrapper ---
# This wrapper is crucial for comparing sum-of-squares values correctly.
# Scipy's least_squares minimizes 0.5 * sum(f_i**2).
# Our chi2 is sum( (sqrt(W_i) * (model_i - data_i))^2 ) = sum(f_i**2).
# So, chi2_scipy = 2 * scipy_res.cost if residual function returns f_i.
def residuals_for_scipy(p_scipy, model_func, t_scipy, y_scipy, weights_scipy_sqrt):
    residuals = model_func(t_scipy, p_scipy) - y_scipy
    if weights_scipy_sqrt is not None:
        return weights_scipy_sqrt * residuals
    return residuals


# --- Test Fixtures ---
@pytest.fixture(name="exp_decay_data_dict")  # Give fixture a distinct name
def exp_decay_data_fixture():  # Use a unique function name
    p_true = np.array([5.0, 2.0, 2.0, 10.0], dtype=np.float64)
    t_data = np.linspace(0.1, 25, 100, dtype=np.float64)  # More points for stability
    y_clean = model_exponential_decay(t_data, p_true)
    return {
        "t": t_data,
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_exponential_decay,
    }


@pytest.fixture(name="polarization_data_dict")  # Give fixture a distinct name
def polarization_data_fixture():  # Use a unique function name
    p_true = np.array([2.0, 3.0, 5.0, 18.0], dtype=np.float64)
    t_data = np.linspace(1.0, 89.0, 100, dtype=np.float64)  # More points
    y_clean = model_polarization(t_data, p_true)
    return {
        "t": t_data,
        "y_clean": y_clean,
        "p_true": p_true,
        "model": model_polarization,
    }


# --- Test Configs ---
# Adjusted tolerances for balance between speed and precision in tests
LMBA_TOL_CONFIG = {"tol_g": 1e-7, "tol_p": 1e-7, "tol_c": 1e-7, "max_iter": 1000}
# Scipy's default ftol, xtol, gtol are 1e-8. max_nfev is number of func evaluations.
SCIPY_TOL_CONFIG = {
    "ftol": 1e-7,
    "xtol": 1e-7,
    "gtol": 1e-7,
    "max_nfev": 2000,
}  # Reduced max_nfev for faster tests if needed


# --- Test Functions ---


# Parametrize with fixture NAMES (strings)
@pytest.mark.parametrize(
    "data_fixture_name", ["exp_decay_data_dict", "polarization_data_dict"]
)
def test_no_noise(
    data_fixture_name, request
):  # Receive fixture name and request fixture
    """Test models with no noise, expects precise parameter recovery."""
    # Get the actual fixture result using request.getfixturevalue
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result  # Unpack dict from fixture

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
        lmba_params["dp_ratio"] = 1e-7  # This was found to be helpful for polarization

    # Run the LMba fit on clean (no-noise) data
    # Pass weights=None for no-noise fits unless your setup dictates otherwise
    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        d["model"], d["t"], d["y_clean"], p_initial, weights=None, **lmba_params
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
    data_fixture_result = request.getfixturevalue(data_fixture_name)
    d = data_fixture_result  # Unpack dict

    # Use a specific seed for reproducibility of noise
    rng = np.random.default_rng(0 if d["model"] == model_exponential_decay else 1)
    # Use a noise level consistent with your previous tests where failures occurred
    noise_std = 0.2 if d["model"] == model_exponential_decay else 0.05

    noise = rng.normal(0, noise_std, size=d["y_clean"].shape).astype(np.float64)
    y_noisy = (d["y_clean"] + noise).astype(np.float64)

    # Use the same initial guess as in the no-noise test or a common one
    p_initial = (d["p_true"] * 0.9).astype(np.float64)
    if d["model"] == model_polarization:
        p_initial = np.array([2.2, 3.3, 5.3, 16.0], dtype=np.float64)

    # Calculate weights based on the noise_std
    weights_arr = np.full_like(y_noisy, 1.0 / (noise_std**2 + _EPS), dtype=np.float64)
    sqrt_weights_arr = np.sqrt(weights_arr)

    lmba_params = LMBA_TOL_CONFIG.copy()
    if d["model"] == model_polarization:
        lmba_params["dp_ratio"] = 1e-7

    # LMba fit
    p_fit_lmba, _, chi2_lmba, iters_lmba, conv_lmba = levenberg_marquardt_core(
        d["model"], d["t"], y_noisy, p_initial, weights=weights_arr, **lmba_params
    )
    assert conv_lmba, (
        f"{d['model'].__name__} (noise) LMba failed: iter={iters_lmba}, chi2={chi2_lmba:.2e}"
    )

    # Scipy fit
    scipy_res = least_squares(
        residuals_for_scipy,
        p_initial,
        args=(d["model"], d["t"], y_noisy, sqrt_weights_arr),  # Pass sqrt_weights
        method="lm",
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res.success, (
        f"{d['model'].__name__} (noise) Scipy failed. Status: {scipy_res.status}, Msg: {scipy_res.message}"
    )
    p_fit_scipy = scipy_res.x
    chi2_scipy = np.sum(scipy_res.fun**2)  # fun already returns weighted residuals

    # Compare results
    # We expect some difference due to noise and algorithm variations
    # >>> COMMENT OUT THE CHI2 COMPARISON AS REQUESTED <<<
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


def test_pixelwise_fitting(exp_decay_data_dict):  # Use the named fixture
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

    # Use a perturbed global initial guess
    p0_global = (d["p_true"] * 0.9).astype(np.float64)

    # Use slightly looser tolerances for pixelwise test to ensure it finishes quickly
    lmba_pixel_params = LMBA_TOL_CONFIG.copy()
    lmba_pixel_params.update(
        {"max_iter": 300, "tol_g": 1e-6, "tol_p": 1e-6, "tol_c": 1e-6}
    )

    p_res, cov_res, chi2_res, n_iter_res, conv_res = levenberg_marquardt_pixelwise(
        d["model"], d["t"], y_data_3d, p0_global, **lmba_pixel_params
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

    # Use strict tolerances for exact solution
    lmba_params_strict = LMBA_TOL_CONFIG.copy()
    lmba_params_strict.update(
        {"tol_g": 1e-9, "tol_p": 1e-9, "tol_c": 1e-9, "max_iter": 50}
    )

    p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
        model_constant, t_data, y_data, p_initial, **lmba_params_strict
    )
    assert conv, (
        f"Singular Jacobian test failed to converge. Iter: {iters}, Chi2: {chi2:.2e}"
    )
    npt.assert_allclose(
        p_fit, p_true, atol=1e-8, err_msg="Singular Jacobian test param mismatch"
    )  # Tighter atol for exact solution
    assert chi2 < 1e-15, (
        f"Singular Jacobian test Chi2 too high: {chi2:.2e}"
    )  # Tighter chi2


def test_weights_effect_vs_scipy(exp_decay_data_dict):  # Use the named fixture
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

    # LMba fit with weights
    p_fit_lmba_w, _, chi2_lmba_w, iters_lmba_w, conv_lmba_w = levenberg_marquardt_core(
        d["model"],
        d["t"],
        y_noisy,
        p_initial_test,
        weights=weights_arr,
        **LMBA_TOL_CONFIG,
    )
    assert conv_lmba_w, (
        f"LMba with weights failed (iters={iters_lmba_w}, chi2={chi2_lmba_w})"
    )

    # Scipy fit with weights
    scipy_res_w = least_squares(
        residuals_for_scipy,
        p_initial_test,
        args=(d["model"], d["t"], y_noisy, sqrt_weights_arr),
        method="lm",
        **SCIPY_TOL_CONFIG,
    )
    assert scipy_res_w.success, (
        f"Scipy with weights failed. Status: {scipy_res_w.status}, Msg: {scipy_res_w.message}"
    )
    p_fit_scipy_w = scipy_res_w.x
    chi2_scipy_w = np.sum(scipy_res_w.fun**2)

    # Compare weighted fits
    # >>> COMMENT OUT THE CHI2 COMPARISON AS REQUESTED <<<
    # npt.assert_allclose(chi2_lmba_w, chi2_scipy_w, rtol=0.2,
    #                     err_msg=f"Weighted chi2 mismatch. LMba: {chi2_lmba_w}, Scipy: {chi2_scipy_w}")

    # Parameters should still be reasonably close if both converged
    npt.assert_allclose(
        p_fit_lmba_w,
        p_fit_scipy_w,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Weighted param mismatch. LMba: {p_fit_lmba_w}, Scipy: {p_fit_scipy_w}",
    )

    # LMba fit NO weights
    p_fit_lmba_nw, _, chi2_lmba_nw, iters_lmba_nw, conv_lmba_nw = (
        levenberg_marquardt_core(
            d["model"], d["t"], y_noisy, p_initial_test, weights=None, **LMBA_TOL_CONFIG
        )
    )
    assert conv_lmba_nw, (
        f"LMba no weights failed (iters={iters_lmba_nw}, chi2={chi2_lmba_nw})"
    )

    # Check that weighted and unweighted fits (from LMba) give different parameters
    diff_sum_params = np.sum(np.abs(p_fit_lmba_w - p_fit_lmba_nw))
    assert diff_sum_params > 1e-3, (
        "LMba: Weighted and unweighted parameters are too close, weights might not have a significant effect."
    )

    # Optionally, check if the chi2 changes meaningfully between weighted and unweighted fits (for LMba)
    unweighted_residuals_at_weighted_fit = d["model"](d["t"], p_fit_lmba_w) - y_noisy
    unweighted_chi2_at_weighted_fit = np.sum(unweighted_residuals_at_weighted_fit**2)

    unweighted_residuals_at_unweighted_fit = d["model"](d["t"], p_fit_lmba_nw) - y_noisy
    unweighted_chi2_at_unweighted_fit = np.sum(
        unweighted_residuals_at_unweighted_fit**2
    )
