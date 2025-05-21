"""
Lumafit: A Numba-accelerated Levenberg-Marquardt Fitting Library.

This library provides functions for performing non-linear least squares
fitting using the Levenberg-Marquardt algorithm, with emphasis on
performance acceleration via Numba JIT compilation and parallel execution.
It is particularly suited for fitting large numbers of curves, such as
pixel-wise fitting in 3D image data.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numba import jit, prange

__version__ = "0.1.0"

# Define epsilon near machine precision for numerical stability
_EPS = np.finfo(float).eps

# References for the Levenberg-Marquardt algorithm implementations:
# [1] Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares.
# [2] Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters.
# [3] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer. (Chapter 10)
# [4] Gavin, H.P. (2020) The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems.

# Define the expected signature for Jacobian functions (both analytical and finite difference)
# A Jacobian function should accept (t, p) and return the Jacobian matrix J (m x n)
# where m is len(t), n is len(p).
JacobianFunc = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
]


@jit(nopython=True, cache=True, fastmath=True)
def _lm_finite_difference_jacobian(
    func: Callable,
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    y_hat: npt.NDArray[np.float64],
    dp_ratio: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """
    Computes Jacobian dy/dp via forward finite differences.

    This is the fallback method used by :func:`levenberg_marquardt_core` if no
    analytical Jacobian is provided. Numba JIT compiled for performance.

    Parameters
    ----------
    func : callable
        The model function `y_hat = func(t, p)`. Must be Numba JIT-compilable.
    t : numpy.ndarray
        Independent variable data (m-element 1D array).
    p : numpy.ndarray
        Current parameter values (n-element 1D array).
    y_hat : numpy.ndarray
        Model evaluation at current `p`, i.e., `func(t, p)`.
    dp_ratio : float, optional
        Fractional increment base for `p` for numerical derivatives.
        Actual step is `dp_ratio * (1 + abs(p))`. Default is 1e-8.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix (m x n).
    """
    m = t.shape[0]
    n = p.shape[0]
    J = np.empty((m, n), dtype=p.dtype)
    p_temp = p.copy()

    # Calculate step size for each parameter for finite differences
    # Based on a fractional change of the parameter's magnitude,
    # or an absolute step if the parameter is near zero.
    h_steps = dp_ratio * (1.0 + np.abs(p))

    # Serial loop over parameters is typically fine here.
    for j in range(n):
        p_j_original = p_temp[j]
        step = h_steps[j]
        if step == 0.0:  # Fallback if step becomes zero
            step = dp_ratio

        # Perturb parameter p[j]
        p_temp[j] = p_j_original + step
        y_plus = func(t, p_temp)  # Evaluate model with perturbed parameter
        p_temp[j] = p_j_original  # Restore parameter

        # Forward difference formula for derivative: (f(x+h) - f(x)) / h
        J[:, j] = (y_plus - y_hat) / step
    return J


@jit(nopython=True, cache=True, fastmath=True)
def levenberg_marquardt_core(
    func: Callable,
    t: npt.NDArray[np.float64],
    y_dat: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    lambda_up_factor: float = 3.0,
    lambda_down_factor: float = 2.0,
    dp_ratio: float = 1e-8,
    weights: npt.NDArray[np.float64] | None = None,
    use_marquardt_damping: bool = True,
    jac_func: JacobianFunc | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int, bool]:
    """
    Core Levenberg-Marquardt algorithm for non-linear least squares curve fitting.

    Implements the standard Levenberg-Marquardt optimization method to find the
    parameters `p` that minimize the sum of squared weighted residuals:
    `sum(weights * (y_dat - func(t, p))**2)`.

    Supports optional weighting and uses a damping strategy (Marquardt-style
    diagonal scaling or Levenberg-style identity matrix scaling). It can utilize
    an analytical Jacobian function if provided, falling back to a
    Numba-accelerated finite difference calculation otherwise.

    This is the core, single-curve fitting routine designed to be JIT compiled
    for performance and used by higher-level functions like
    :func:`levenberg_marquardt_pixelwise`.

    Parameters
    ----------
    func : callable
        The model function `y_hat = func(t, p)`. It must be a Numba JIT-compilable
        function that accepts a 1D array `t` (independent variable) and a 1D
        array `p` (parameters), and returns a 1D array `y_hat` (model output)
        of the same length as `t`.
    t : numpy.ndarray
        Independent variable data (m-element 1D array, float64 dtype).
    y_dat : numpy.ndarray
        Dependent variable (experimental) data (m-element 1D array, float64 dtype).
    p0 : numpy.ndarray
        Initial guess for the parameters `p` (n-element 1D array, float64 dtype).
    max_iter : int, default=100
        Maximum number of iterations for the optimization loop.
    tol_g : float, default=1e-8
        Convergence tolerance on the maximum absolute component of the weighted
        gradient `J^T W dy`. If the maximum absolute gradient is less than this
        value, the algorithm is considered converged.
    tol_p : float, default=1e-8
        Convergence tolerance on the relative change in the parameter vector
        between successive accepted steps. Convergence is met if the maximum
        relative step `|dp[i]| / (|p_try[i]| + EPS)` is less than this value
        for all parameters.
    tol_c : float, default=1e-8
        Convergence tolerance on the relative change in the weighted Chi-squared
        value between successive accepted steps. Convergence is met if
        `(Chi2_prev - Chi2_current) / Chi2_prev` is less than this value
        (for Chi2_prev > EPS).
    lambda_0_factor : float, default=1e-2
        Initial scaling factor for the damping parameter `lambda`. The initial
        `lambda` is typically set based on this factor and the diagonal of
        `J^T W J`.
    lambda_up_factor : float, default=3.0
        Factor by which the damping parameter `lambda` is increased when a
        trial step is rejected (i.e., increases Chi-squared).
    lambda_down_factor : float, default=2.0
        Factor by which the damping parameter `lambda` is decreased when an
        accepted step results in a reduction in Chi-squared.
    dp_ratio : float, default=1e-8
        Fractional increment base used for calculating the finite difference
        Jacobian if `jac_func` is None. The step size for parameter `p[j]`
        is `dp_ratio * (1 + abs(p[j]))`.
    weights : numpy.ndarray or None, optional, default=None
        Weights array (m-element 1D array, float64 dtype). The algorithm
        minimizes the sum of weighted squared residuals `sum(weights * (y_dat - func(t, p))**2)`.
        If None, uniform weights of 1.0 are used. Must be Numba compatible.
    use_marquardt_damping : bool, optional, default=True
        If True, the Marquardt damping scheme is used where `lambda` scales
        the diagonal elements of `J^T W J`. If False, the Levenberg scheme is used
        where `lambda` scales the identity matrix, adding `lambda` to the diagonal.
        Marquardt damping is often preferred as it is scale-invariant with respect
        to parameter scaling.
    jac_func : callable or None, optional, default=None
        Analytical Jacobian function `J = jac_func(t, p)`. It must be a
        Numba JIT-compilable function that accepts a 1D array `t` and a 1D
        array `p`, and returns the (m x n) Jacobian matrix as a 2D NumPy array
        (float64 dtype), where m is the number of data points and n is the
        number of parameters. If None, the Jacobian is computed using
        finite differences via :func:`_lm_finite_difference_jacobian`.

    Returns
    -------
    p_fit : numpy.ndarray
        The fitted parameter values (n-element 1D array, float64 dtype).
    cov_p : numpy.ndarray
        The estimated covariance matrix of the fitted parameters (n x n 2D array,
        float64 dtype). This is computed as `(J^T W J)^-1`. It contains `np.inf`
        or `np.nan` if the matrix is singular or cannot be inverted (e.g.,
        insufficient degrees of freedom, m <= n).
    chi2 : float
        The final weighted Chi-squared value (`sum(weights * residuals**2)`).
    n_iter_final : int
        The number of iterations performed until convergence or reaching `max_iter`.
    converged : bool
        True if the algorithm converged within `max_iter` according to the
        specified tolerances (`tol_g`, `tol_p`, or `tol_c`), False otherwise.

    Notes
    -----
    The function is decorated with ``@jit(nopython=True, cache=True, fastmath=True)``
    for Numba acceleration. Ensure that ``func``, ``jac_func`` (if provided), and
    all operations within them are Numba-compatible.

    Uses a small epsilon value (``_EPS``) derived from machine precision for
    numerical stability, e.g., in checks for near-zero denominators or matrix
    diagonals.

    Convergence is checked *after* a successful step (where Chi-squared decreases).

    The covariance matrix ``(J^T W J)^-1`` is a standard output of LM. If weights ``W``
    represent ``1/sigma_i^2`` (inverse variances), this is the parameter covariance.
    If weights are uniform (unweighted least squares), this needs to be scaled
    by the reduced Chi-squared (``chi2 / (m - n)``) to estimate the parameter
    covariance matrix assuming independent, identically distributed errors. This
    scaling is **not** performed by this core function; it returns ``(J^T W J)^-1`` directly.

    For a detailed description of the algorithm and its convergence properties,
    see :cite:`levenberg1944method`, :cite:`marquardt1963algorithm`,
    and :cite:`nocedal2006numerical`. The implementation loosely follows
    concepts described by :cite:`gavin2020levenberg`.

    See Also
    --------
    :func:`levenberg_marquardt_pixelwise` : Applies this core algorithm to 3D pixel data.
    :func:`_lm_finite_difference_jacobian` : The internal function used for FD Jacobian when `jac_func` is None.
    """
    m = t.shape[0]
    n = p0.shape[0]
    p = p0.copy()

    if weights is None:
        W_arr = np.ones(m, dtype=y_dat.dtype)
    else:
        W_arr = weights.copy()

    y_hat = func(t, p)
    J: npt.NDArray[np.float64]
    if jac_func is not None:
        J = jac_func(t, p)
    else:
        J = _lm_finite_difference_jacobian(func, t, p, y_hat, dp_ratio)

    W_dy = W_arr * (y_dat - y_hat)
    if W_arr.ndim == 1:
        W_J = W_arr[:, np.newaxis] * J
    else:
        W_J = W_arr * J

    JtWJ = J.T @ W_J
    JtWdy = J.T @ W_dy
    chi2 = np.sum(W_dy**2)

    current_max_grad = np.max(np.abs(JtWdy))
    if current_max_grad < tol_g:
        converged = True
        n_iter_final = 0
        final_cov_p = np.full((n, n), np.nan, dtype=p.dtype)
        try:
            if (m - n > 0) and np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ).astype(np.float64)
        except Exception:
            pass
        return p, final_cov_p, chi2, n_iter_final, converged
    else:
        converged = False

    lambda_val: float
    if use_marquardt_damping:
        diag_JtWJ_init = np.diag(JtWJ)
        diag_JtWJ_init_stable = diag_JtWJ_init + _EPS * (diag_JtWJ_init == 0.0)
        max_diag_val = np.max(diag_JtWJ_init_stable)
        lambda_val = (
            lambda_0_factor * max_diag_val if max_diag_val > _EPS else lambda_0_factor
        )
    else:
        lambda_val = lambda_0_factor

    if lambda_val <= 0.0 or not np.isfinite(lambda_val):
        lambda_val = 1e-2

    n_iter_final = 0
    for k_iter_loop in range(max_iter):
        n_iter_final = k_iter_loop + 1

        # --- Store Chi-squared value at the beginning of the current iteration ---
        chi2_at_iter_start = chi2  # <<< ADD THIS LINE BACK

        # Construct the augmented Hessian matrix 'A'
        A: npt.NDArray[np.float64]
        if use_marquardt_damping:
            diag_JtWJ = np.diag(JtWJ)
            diag_JtWJ_stable = diag_JtWJ + _EPS * (diag_JtWJ == 0.0)
            A = JtWJ + lambda_val * np.diag(diag_JtWJ_stable)
        else:
            A = JtWJ + lambda_val * np.eye(n, dtype=JtWJ.dtype)

        # Solve for the parameter step 'dp'
        dp_step: npt.NDArray[np.float64]
        try:
            dp_step = np.linalg.solve(A, JtWdy)
        except Exception:
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

        if not np.all(np.isfinite(dp_step)):
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

        p_try = p + dp_step
        y_hat_try = func(t, p_try)

        if not np.all(np.isfinite(y_hat_try)):
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

        dy_try = y_dat - y_hat_try
        W_dy_try = W_arr * dy_try
        chi2_try = np.sum(W_dy_try**2)

        # Decision step: Check if the trial step improved Chi-squared value
        if chi2_try < chi2_at_iter_start:  # Step accepted
            lambda_val /= lambda_down_factor
            lambda_val = np.maximum(lambda_val, 1e-15)

            p = p_try
            chi2 = chi2_try
            y_hat = y_hat_try

            # Recalculate Jacobian and JtW components at the new accepted parameters
            if jac_func is not None:
                J = jac_func(t, p)
            else:
                J = _lm_finite_difference_jacobian(func, t, p, y_hat, dp_ratio)

            if W_arr.ndim == 1:
                W_J = W_arr[:, np.newaxis] * J
            else:
                W_J = W_arr * J
            JtWJ = J.T @ W_J
            JtWdy = J.T @ (W_arr * (y_dat - y_hat))

            # --- Check Convergence Criteria AFTER a successful step ---
            current_max_grad_after_step = np.max(np.abs(JtWdy))
            if current_max_grad_after_step < tol_g:
                converged = True
                break

            dChi2_this_step = chi2_at_iter_start - chi2
            rel_dChi2_this_step = (
                dChi2_this_step / chi2_at_iter_start
                if chi2_at_iter_start > _EPS
                else 0.0
            )
            if (
                n_iter_final > 1
                and chi2_at_iter_start > _EPS
                and rel_dChi2_this_step < tol_c
            ):
                converged = True
                break

            rel_dp_for_step = np.abs(dp_step) / (np.abs(p_try) + _EPS)
            max_rel_dp_this_step = np.max(rel_dp_for_step)
            if max_rel_dp_this_step < tol_p:
                converged = True
                break

        else:  # Step rejected
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

    # --- End of Main Loop ---
    if not converged and n_iter_final == max_iter:
        current_max_grad_at_max_iter = np.max(np.abs(JtWdy))
        if current_max_grad_at_max_iter < tol_g:
            converged = True

    final_cov_p = np.full((n, n), np.inf, dtype=p.dtype)
    try:
        dof = m - n
        if dof > 0:
            if np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ).astype(np.float64)
    except Exception:
        pass

    return p, final_cov_p, chi2, n_iter_final, converged


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def levenberg_marquardt_pixelwise(
    func: Callable,
    t: npt.NDArray[np.float64],
    y_dat_3d: npt.NDArray[np.float64],
    p0_global: npt.NDArray[np.float64],
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    lambda_up_factor: float = 3.0,
    lambda_down_factor: float = 2.0,
    dp_ratio: float = 1e-8,
    weights_1d: npt.NDArray[np.float64] | None = None,
    use_marquardt_damping: bool = True,
    jac_func: JacobianFunc | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
]:
    """
    Applies Levenberg-Marquardt fitting pixel-wise to 3D data.

    Numba JIT compiled with ``parallel=True`` for performing fits on multiple
    pixels concurrently using ``numba.prange``. Each pixel's curve
    ``y_dat_3d[r, c, :]`` is fitted independently using :func:`levenberg_marquardt_core`.

    Parameters
    ----------
    func : callable
        Model function `y_hat = func(t, p)`. Must be Numba JIT-compilable
        and accept `t` (1D array) and `p` (1D array) as arguments, returning
        a 1D array of model values.
    t : numpy.ndarray
        Independent variable data, common for all pixels (m-element 1D array).
    y_dat_3d : numpy.ndarray
        Dependent variable data (rows x cols x m_depth_points 3D array).
        Each `y_dat_3d[r, c, :]` is a curve to be fitted.
    p0_global : numpy.ndarray
        Global initial guess for parameters (n-element 1D array),
        used for each pixel's fit.
    max_iter, tol_g, ..., use_marquardt_damping :
        Parameters for :func:`levenberg_marquardt_core` (see its docstring).
    dp_ratio : float, optional
        Step size ratio for finite difference Jacobian calculation if `jac_func` is None.
        Default is 1e-8.
    weights_1d : numpy.ndarray or None, optional
        1D array of weights (m-element), applied identically to each pixel's fit.
        If None (default), uniform weights (1.0) are used. Must be Numba compatible.
    jac_func : callable or None, optional
        Analytical Jacobian function `J = jac_func(t, p)`. Must be Numba
        JIT-compilable and accept `t` (1D array) and `p` (1D array) as
        arguments, returning the (m x n) Jacobian as a 2D NumPy array. If
        None (default), the finite difference Jacobian is calculated internally.

    Returns
    -------
    tuple
        Contains 3D/2D arrays corresponding to the outputs of :func:`levenberg_marquardt_core` for each pixel:
            - p_results : numpy.ndarray (rows x cols x n_params)
                Fitted parameters for each pixel. Contains `np.nan` for skipped pixels.
            - cov_p_results : numpy.ndarray (rows x cols x n_params x n_params)
                Covariance matrices for each pixel. Contains `np.inf`/`np.nan` for
                non-calculable/skipped pixels.
            - chi2_results : numpy.ndarray (rows x cols)
                Final weighted Chi-squared values for each pixel. Contains `np.nan`
                for skipped pixels.
            - n_iter_results : numpy.ndarray (rows x cols, dtype=np.int_)
                Number of iterations for each pixel. Contains 0 for skipped pixels.
            - converged_results : numpy.ndarray (rows x cols, dtype=bool)
                Convergence status for each pixel. Contains False for skipped pixels.

    """
    rows = y_dat_3d.shape[0]
    cols = y_dat_3d.shape[1]
    num_params = p0_global.shape[0]

    # Pre-allocate output arrays with default values for skipped pixels
    p_results = np.full((rows, cols, num_params), np.nan, dtype=p0_global.dtype)
    cov_p_results = np.full(
        (rows, cols, num_params, num_params), np.nan, dtype=p0_global.dtype
    )
    chi2_results = np.full((rows, cols), np.nan, dtype=p0_global.dtype)
    n_iter_results = np.zeros((rows, cols), dtype=np.int_)
    converged_results = np.zeros((rows, cols), dtype=np.bool_)  # Use bool for flags

    # Loop over pixels in parallel using numba.prange
    # Flatten the 2D pixel indices (row, col) into a single 1D index for prange
    for flat_idx in prange(rows * cols):
        # Convert flat index back to 2D indices
        r = flat_idx // cols  # Integer division gets the row index
        c = flat_idx % cols  # Modulo gets the column index

        # Extract data for the current pixel
        y_pixel_data = y_dat_3d[r, c, :]

        # Skip pixels with NaN data (e.g., masked pixels)
        if np.any(np.isnan(y_pixel_data)):
            continue  # Output arrays already initialized to indicate skipped/invalid fit

        # Create a copy of the initial guess for the core LM function.
        # Although LM core copies p0 internally, explicitly copying here
        # ensures thread-local data if p0_global were mutable (which it isn't, but good practice).
        p0_pixel = p0_global.copy()

        # Call the core LM algorithm for the current pixel's data
        p_fit, cov_p, chi2_val, iters, conv_flag = levenberg_marquardt_core(
            func,
            t,
            y_pixel_data,
            p0_pixel,  # Pass the pixel-specific initial guess
            max_iter=max_iter,
            tol_g=tol_g,
            tol_p=tol_p,
            tol_c=tol_c,
            lambda_0_factor=lambda_0_factor,
            lambda_up_factor=lambda_up_factor,
            lambda_down_factor=lambda_down_factor,
            dp_ratio=dp_ratio,
            weights=weights_1d,  # Pass the 1D weights array (same for all pixels)
            use_marquardt_damping=use_marquardt_damping,
            jac_func=jac_func,  # Pass the optional jac_func
        )

        # Store the results for the current pixel back into the pre-allocated output arrays
        p_results[r, c, :] = p_fit
        cov_p_results[r, c, :, :] = cov_p
        chi2_results[r, c] = chi2_val
        n_iter_results[r, c] = iters
        converged_results[r, c] = conv_flag

    return p_results, cov_p_results, chi2_results, n_iter_results, converged_results
