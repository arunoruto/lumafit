"""
LMba: A Numba-accelerated Levenberg-Marquardt Fitting Library
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numba import jit, prange  # Explicitly add nopython=True for clarity

# Define eps near machine precision for numerical stability
_EPS = np.finfo(float).eps

# References for Levenberg-Marquardt algorithm:
# [1] Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares".
# [2] Marquardt, D. W. (1963). "An algorithm for least-squares estimation of nonlinear parameters".
# [3] Nocedal, J., & Wright, S. (2006). "Numerical optimization". Springer. (Chapter 10)
# [4] Gavin, H.P. (2020) "The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems."
#     (Often cited, good practical overview)


@jit(cache=True, fastmath=True)
def _lm_finite_difference_jacobian(
    func: Callable,
    t: npt.NDArray[np.float64],  # CHANGED from np.float_
    p: npt.NDArray[np.float64],  # CHANGED from np.float_
    y_hat: npt.NDArray[np.float64],  # CHANGED from np.float_
    dp_ratio: float = 1e-8,
) -> npt.NDArray[np.float64]:  # CHANGED from np.float_
    """Computes Jacobian dy/dp via forward finite differences.

    Numba JIT compiled for performance.

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
    J = np.empty((m, n), dtype=p.dtype)  # p.dtype will be float64
    p_temp = p.copy()

    h_steps = dp_ratio * (1.0 + np.abs(p))

    for j in range(n):
        p_j_original = p_temp[j]
        step = h_steps[j]
        if step == 0.0:
            step = dp_ratio

        p_temp[j] = p_j_original + step
        y_plus = func(t, p_temp)
        p_temp[j] = p_j_original

        J[:, j] = (y_plus - y_hat) / step
    return J


@jit(cache=True, fastmath=True)
def levenberg_marquardt_core(
    func: Callable,
    t: npt.NDArray[np.float64],  # CHANGED
    y_dat: npt.NDArray[np.float64],  # CHANGED
    p0: npt.NDArray[np.float64],  # CHANGED
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    lambda_up_factor: float = 3.0,
    lambda_down_factor: float = 2.0,
    dp_ratio: float = 1e-8,
    weights: npt.NDArray[np.float64] | None = None,  # CHANGED
    use_marquardt_damping: bool = True,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], float, int, bool
]:  # CHANGED (for array returns)
    """Core Levenberg-Marquardt algorithm for non-linear least squares.
    # ... (rest of docstring unchanged)
    """
    m = t.shape[0]
    n = p0.shape[0]
    p = p0.copy()

    if weights is None:
        W_arr = np.ones(m, dtype=y_dat.dtype)  # y_dat.dtype will be float64
    else:
        W_arr = weights.copy()

    y_hat = func(t, p)
    dy = y_dat - y_hat
    J = _lm_finite_difference_jacobian(func, t, p, y_hat, dp_ratio)

    W_dy = W_arr * dy
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
        final_cov_p = np.full((n, n), np.nan, dtype=p.dtype)  # ensure dtype consistency
        try:
            if m - n > 0 and np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ)
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
        chi2_at_iter_start = chi2

        A: npt.NDArray[np.float64]  # Ensure consistent type
        if use_marquardt_damping:
            diag_JtWJ = np.diag(JtWJ)
            diag_JtWJ_stable = diag_JtWJ + _EPS * (diag_JtWJ == 0.0)
            A = JtWJ + lambda_val * np.diag(diag_JtWJ_stable)
        else:
            A = JtWJ + lambda_val * np.eye(n, dtype=JtWJ.dtype)

        dp_step: npt.NDArray[np.float64]  # Ensure consistent type
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

        rel_dp_for_step = np.abs(dp_step) / (np.abs(p) + _EPS)
        max_rel_dp_this_step = np.max(rel_dp_for_step)

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

        if chi2_try < chi2_at_iter_start:
            lambda_val /= lambda_down_factor
            lambda_val = np.maximum(lambda_val, 1e-15)

            p = p_try
            chi2 = chi2_try
            y_hat = y_hat_try

            J = _lm_finite_difference_jacobian(func, t, p, y_hat, dp_ratio)
            if W_arr.ndim == 1:
                W_J = W_arr[:, np.newaxis] * J
            else:
                W_J = W_arr * J
            JtWJ = J.T @ W_J
            JtWdy = J.T @ (W_arr * (y_dat - y_hat))

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

            if max_rel_dp_this_step < tol_p:
                converged = True
                break
        else:
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

    if not converged and n_iter_final == max_iter:
        current_max_grad_at_max_iter = np.max(np.abs(JtWdy))
        if current_max_grad_at_max_iter < tol_g:
            converged = True

    final_cov_p = np.full((n, n), np.inf, dtype=p.dtype)  # Ensure dtype
    try:
        dof = m - n
        if dof > 0:
            if np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ)
    except Exception:
        pass
    return p, final_cov_p, chi2, n_iter_final, converged


@jit(parallel=True, cache=True, fastmath=True)
def levenberg_marquardt_pixelwise(
    func: Callable,
    t: npt.NDArray[np.float64],  # CHANGED
    y_dat_3d: npt.NDArray[np.float64],  # CHANGED
    p0_global: npt.NDArray[np.float64],  # CHANGED
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    lambda_up_factor: float = 10.0,
    lambda_down_factor: float = 10.0,
    dp_ratio: float = 1e-8,
    weights_1d: npt.NDArray[np.float64] | None = None,  # CHANGED
    use_marquardt_damping: bool = True,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
]:  # CHANGED (for array returns)
    """Applies Levenberg-Marquardt fitting pixel-wise to 3D data.
    # ... (rest of docstring unchanged)
    """
    rows = y_dat_3d.shape[0]
    cols = y_dat_3d.shape[1]
    num_params = p0_global.shape[0]

    p_results = np.full((rows, cols, num_params), np.nan, dtype=p0_global.dtype)
    cov_p_results = np.full(
        (rows, cols, num_params, num_params), np.nan, dtype=p0_global.dtype
    )
    chi2_results = np.full((rows, cols), np.nan, dtype=p0_global.dtype)
    n_iter_results = np.zeros((rows, cols), dtype=np.int32)
    converged_results = np.zeros((rows, cols), dtype=np.bool_)

    for flat_idx in prange(rows * cols):
        r = flat_idx // cols
        c = flat_idx % cols

        y_pixel_data = y_dat_3d[r, c, :]

        if np.any(np.isnan(y_pixel_data)):
            continue

        p0_pixel = p0_global.copy()

        p_fit, cov_p, chi2_val, iters, conv_flag = levenberg_marquardt_core(
            func,
            t,
            y_pixel_data,
            p0_pixel,
            max_iter=max_iter,
            tol_g=tol_g,
            tol_p=tol_p,
            tol_c=tol_c,
            lambda_0_factor=lambda_0_factor,
            lambda_up_factor=lambda_up_factor,
            lambda_down_factor=lambda_down_factor,
            dp_ratio=dp_ratio,
            weights=weights_1d,
            use_marquardt_damping=use_marquardt_damping,
        )

        p_results[r, c, :] = p_fit
        cov_p_results[r, c, :, :] = cov_p
        chi2_results[r, c] = chi2_val
        n_iter_results[r, c] = iters
        converged_results[r, c] = conv_flag

    return p_results, cov_p_results, chi2_results, n_iter_results, converged_results
