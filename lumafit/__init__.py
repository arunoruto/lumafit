"""
Lumafit: A Numba-accelerated Levenberg-Marquardt Fitting Library
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numba import jit, prange

_EPS = np.finfo(float).eps

# References for Levenberg-Marquardt algorithm:
# [1] Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares".
# [2] Marquardt, D. W. (1963). "An algorithm for least-squares estimation of nonlinear parameters".
# [3] Nocedal, J., & Wright, S. (2006). "Numerical optimization". Springer. (Chapter 10)
# [4] Gavin, H.P. (2020) "The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems."
#     (Often cited, good practical overview)

# Define the expected signature for Jacobian functions (both analytical and finite difference)
# J(t, p) -> J (m x n)
# where m is len(t), n is len(p)
JacobianFunc = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
]


@jit(nopython=True, cache=True, fastmath=True)
def _lm_finite_difference_jacobian(
    func: Callable,
    t: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    y_hat: npt.NDArray[np.float64],  # Note: finite difference also needs y_hat
    dp_ratio: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """Computes Jacobian dy/dp via forward finite differences.

    This is the fallback method used by `levenberg_marquardt_core` if no
    analytical Jacobian is provided. Numba JIT compiled for performance.

    Parameters
    ----------
    func : callable
        The model function `y_hat = func(t, p)`.
    t : numpy.ndarray
        Independent variable data (m-element 1D array).
    p : numpy.ndarray
        Current parameter values (n-element 1D array).
    y_hat : numpy.ndarray
        Model evaluation at current `p`, i.e., `func(t, p)`.
    dp_ratio : float, optional
        Fractional increment of `p` for numerical derivatives.
        Default is 1e-8.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix (m x n).
    """
    m = t.shape[0]
    n = p.shape[0]
    J = np.empty((m, n), dtype=p.dtype)
    p_temp = p.copy()

    h_steps = dp_ratio * (1.0 + np.abs(p))

    for j in range(n):
        p_j_original = p_temp[j]
        step = h_steps[j]
        if step == 0.0:
            step = dp_ratio

        p_temp[j] = p_j_original + step
        y_plus = func(t, p_temp)
        p_temp[j] = p_j_original  # Restore parameter

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
    jac_func: JacobianFunc | None = None,  # New optional analytical Jacobian function
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int, bool]:
    """Core Levenberg-Marquardt algorithm for non-linear least squares.
    # ... (docstring remains the same as before, referencing jac_func) ...
    """
    m = t.shape[0]
    n = p0.shape[0]
    p = p0.copy()

    if weights is None:
        W_arr = np.ones(m, dtype=y_dat.dtype)
    else:
        W_arr = weights.copy()

    # Initial model evaluation
    y_hat = func(t, p)

    # Initial Jacobian calculation: Use analytical if provided, else finite difference
    J: npt.NDArray[np.float64]
    if jac_func is not None:
        J = jac_func(t, p)  # Eq. 6 in Gavin [4]
    else:
        J = _lm_finite_difference_jacobian(
            func, t, p, y_hat, dp_ratio
        )  # Eq. 7 in Gavin [4]

    # Weighted quantities
    W_dy = W_arr * (y_dat - y_hat)  # Residuals: y_data - y_model (Eq. 1 in Gavin [4])

    # Apply weights column-wise to Jacobian: W_J[i,j] = W_arr[i] * J[i,j]
    if W_arr.ndim == 1:
        W_J = W_arr[:, np.newaxis] * J
    else:  # Should not happen if weights is 1D or None
        W_J = W_arr * J

    # Approx. Hessian: JtWJ = J^T * W * J (Eq. 10 in Gavin [4])
    JtWJ = J.T @ W_J
    # Gradient term (right-hand side): JtWdy = J^T * W * dy (Eq. 10 in Gavin [4])
    JtWdy = J.T @ W_dy
    # Chi-squared error: sum( (W * dy)^2 )
    chi2 = np.sum(W_dy**2)

    current_max_grad = np.max(np.abs(JtWdy))
    if current_max_grad < tol_g:  # Check initial gradient
        converged = True
        n_iter_final = 0
        final_cov_p = np.full((n, n), np.nan, dtype=p.dtype)
        try:
            if m - n > 0 and np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ)
        except Exception:
            pass
        return p, final_cov_p, chi2, n_iter_final, converged
    else:
        converged = False

    # Initialize damping parameter lambda (mu in some texts)
    lambda_val: float
    if use_marquardt_damping:
        diag_JtWJ_init = np.diag(JtWJ)
        diag_JtWJ_init_stable = diag_JtWJ_init + _EPS * (diag_JtWJ_init == 0.0)
        max_diag_val = np.max(diag_JtWJ_init_stable)
        lambda_val = (
            lambda_0_factor * max_diag_val if max_diag_val > _EPS else lambda_0_factor
        )
    else:  # Levenberg style: lambda is used directly
        lambda_val = lambda_0_factor

    if lambda_val <= 0.0 or not np.isfinite(lambda_val):
        lambda_val = 1e-2  # Fallback initial lambda

    n_iter_final = 0
    for k_iter_loop in range(max_iter):
        n_iter_final = k_iter_loop + 1
        chi2_at_iter_start = chi2

        # Augmented Hessian A = (JtWJ + lambda * D) (Eq. 15/16 in Gavin [4])
        # D = diag(JtWJ) for Marquardt, D = I for Levenberg
        A: npt.NDArray[np.float64]
        if use_marquardt_damping:
            diag_JtWJ = np.diag(JtWJ)
            diag_JtWJ_stable = diag_JtWJ + _EPS * (diag_JtWJ == 0.0)
            A = JtWJ + lambda_val * np.diag(diag_JtWJ_stable)
        else:  # Levenberg: D = I (identity matrix)
            A = JtWJ + lambda_val * np.eye(n, dtype=JtWJ.dtype)

        # Solve for parameter step: A * dp = JtWdy (Eq. 15 in Gavin [4])
        dp_step: npt.NDArray[np.float64]
        try:
            dp_step = np.linalg.solve(A, JtWdy)
        except Exception:  # Catch LinAlgError (singular matrix) or others
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)  # Cap lambda
            if lambda_val > 1e12:  # If lambda becomes excessively large
                converged = False
                break  # Assume stuck
            continue  # Try next iteration with increased lambda

        if not np.all(np.isfinite(dp_step)):
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

        # Relative change in parameters for this trial step
        rel_dp_for_step = np.abs(dp_step) / (np.abs(p) + _EPS)
        max_rel_dp_this_step = np.max(rel_dp_for_step)

        p_try = p + dp_step
        y_hat_try = func(t, p_try)

        if not np.all(np.isfinite(y_hat_try)):  # Model unstable with p_try
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

        dy_try = y_dat - y_hat_try
        W_dy_try = W_arr * dy_try
        chi2_try = np.sum(W_dy_try**2)

        # Check if the trial step improved Chi-squared
        if chi2_try < chi2_at_iter_start:  # Step accepted
            lambda_val /= lambda_down_factor
            lambda_val = np.maximum(
                lambda_val, 1e-15
            )  # Prevent lambda from being too small

            # Update parameters and related quantities
            p = p_try
            chi2 = chi2_try
            y_hat = y_hat_try
            # dy = dy_try # Not strictly needed further unless used in gain ratio
            # W_dy = W_dy_try # Not strictly needed further

            # Recalculate Jacobian: Use analytical if provided, else finite difference
            if jac_func is not None:
                J = jac_func(t, p)  # Eq. 6 in Gavin [4]
            else:
                J = _lm_finite_difference_jacobian(
                    func, t, p, y_hat, dp_ratio
                )  # Eq. 7 in Gavin [4]

            if W_arr.ndim == 1:
                W_J = W_arr[:, np.newaxis] * J
            else:
                W_J = W_arr * J  # Should not happen
            JtWJ = J.T @ W_J
            JtWdy = J.T @ (
                W_arr * (y_dat - y_hat)
            )  # Recompute JtWdy with current y_hat

            # --- Convergence Checks After Successful Step ---
            current_max_grad_after_step = np.max(np.abs(JtWdy))
            if current_max_grad_after_step < tol_g:
                converged = True
                break

            dChi2_this_step = chi2_at_iter_start - chi2  # Positive value
            # Relative change in Chi-squared
            rel_dChi2_this_step = (
                dChi2_this_step / chi2_at_iter_start
                if chi2_at_iter_start > _EPS
                else 0.0
            )
            # Check after first iter, as first step can have large relative change
            if (
                n_iter_final > 1
                and chi2_at_iter_start > _EPS
                and rel_dChi2_this_step < tol_c
            ):
                converged = True
                break

            # Check parameter change that *led to this accepted state*
            if max_rel_dp_this_step < tol_p:
                converged = True
                break
        else:  # Step rejected
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)  # Cap lambda
            if lambda_val > 1e12:  # If lambda grew excessively, assume stuck
                converged = False
                break
            continue  # Essential: try again with new lambda_val

    # --- End of Loop ---
    # Final convergence check if max_iter reached without explicit convergence
    if not converged and n_iter_final == max_iter:
        current_max_grad_at_max_iter = np.max(np.abs(JtWdy))
        if current_max_grad_at_max_iter < tol_g:
            converged = True  # Converged on gradient at the very last iteration

    # Calculate final covariance matrix
    # cov_p = (J^T * W * J)^-1 (Inverse of approx. Hessian)
    # Scale by reduced chi-squared (chi2 / (m-n)) if weights are relative, not absolute.
    # Here, we provide the unscaled covariance.
    final_cov_p = np.full(
        (n, n), np.inf, dtype=p.dtype
    )  # Default to Inf if not calculable
    try:
        dof = m - n  # Degrees of freedom
        if dof > 0:
            if np.any(np.abs(JtWJ) > _EPS):  # Ensure JtWJ is not all zeros
                final_cov_p = np.linalg.inv(JtWJ)
    except Exception:  # Catch LinAlgError or other issues
        pass  # final_cov_p remains np.inf
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
    weights_1d: npt.NDArray[np.float64]
    | None = None,  # Weights are 1D, applied to each pixel
    use_marquardt_damping: bool = True,
    jac_func: JacobianFunc | None = None,  # New optional analytical Jacobian function
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
]:
    """Applies Levenberg-Marquardt fitting pixel-wise to 3D data.
    # ... (docstring remains the same as before, referencing jac_func and dp_ratio) ...
    """
    rows = y_dat_3d.shape[0]
    cols = y_dat_3d.shape[1]
    num_params = p0_global.shape[0]

    # Pre-allocate output arrays
    p_results = np.full((rows, cols, num_params), np.nan, dtype=p0_global.dtype)
    cov_p_results = np.full(
        (rows, cols, num_params, num_params), np.nan, dtype=p0_global.dtype
    )
    chi2_results = np.full((rows, cols), np.nan, dtype=p0_global.dtype)
    n_iter_results = np.zeros((rows, cols), dtype=np.int32)
    converged_results = np.zeros((rows, cols), dtype=np.bool_)

    # Loop over pixels in parallel
    # prange requires a simple 1D loop. Flatten the 2D pixel indices.
    for flat_idx in prange(rows * cols):
        r = flat_idx // cols  # Convert flat index back to row index
        c = flat_idx % cols  # Convert flat index back to col index

        y_pixel_data = y_dat_3d[r, c, :]

        # Skip pixels with NaN data, as fitting is not possible
        if np.any(np.isnan(y_pixel_data)):
            continue  # Output arrays remain NaN/0/False for this pixel

        # Each thread needs its own copy of p0 to avoid race conditions if p0 were modified
        p0_pixel = p0_global.copy()

        # Call the core LM algorithm for the current pixel
        p_fit, cov_p, chi2_val, iters, conv_flag = levenberg_marquardt_core(
            func,
            t,
            y_pixel_data,
            p0_pixel,  # Pass the copy
            max_iter=max_iter,
            tol_g=tol_g,
            tol_p=tol_p,
            tol_c=tol_c,
            lambda_0_factor=lambda_0_factor,
            lambda_up_factor=lambda_up_factor,
            lambda_down_factor=lambda_down_factor,
            dp_ratio=dp_ratio,
            weights=weights_1d,  # Pass the 1D weights array
            use_marquardt_damping=use_marquardt_damping,
            jac_func=jac_func,  # Pass the optional jac_func
        )

        # Store results for the current pixel
        p_results[r, c, :] = p_fit
        cov_p_results[r, c, :, :] = cov_p
        chi2_results[r, c] = chi2_val
        n_iter_results[r, c] = iters
        converged_results[r, c] = conv_flag

    return p_results, cov_p_results, chi2_results, n_iter_results, converged_results
