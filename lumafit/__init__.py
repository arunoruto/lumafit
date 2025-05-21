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

# Define epsilon near machine precision for numerical stability
_EPS = np.finfo(float).eps

# References for the Levenberg-Marquardt algorithm implementations:
# [1] Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares.
# [2] Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters.
# [3] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer. (Chapter 10)
# [4] Gavin, H.P. (2020) The Levenberg-Marquardt method for nonlinear least squares curve-fitting problems.

# Define the expected signature for model and Jacobian functions:
# - Model function: func(p, t, *args) -> y_hat (or residuals)
# - Jacobian function: jac_func(p, t, *args) -> J (m x n)
# The `p` is always the 1D numpy array of parameters being optimized.
# The `t` is always the 1D numpy array of independent variable data.
# The `args` tuple contains all *other* additional positional arguments.
#
# Note: Callable type hints for `*args` are tricky in Python. We use general types
# and rely on docstrings for clarity that the function must accept `(p, t, *args_tuple)`.
# The Numba JIT-compiled functions must be defined with `*args`.
ModelFunc = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], tuple], npt.NDArray[np.float64]
]
JacobianFunc = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], tuple], npt.NDArray[np.float64]
]


@jit(nopython=True, cache=True, fastmath=True)
def _lm_finite_difference_jacobian(
    func: ModelFunc,
    p: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    y_hat: npt.NDArray[
        np.float64
    ],  # This is actually model output or current residuals
    args: tuple = (),
    dp_ratio: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """
    Computes Jacobian dy/dp via forward finite differences.

    This is the fallback method used by :func:`levenberg_marquardt_core` if no
    analytical Jacobian is provided. Numba JIT compiled for performance.

    Parameters
    ----------
    func : callable
        The model function `y_hat = func(p, t, *args)`. Must be Numba JIT-compilable.
        It should accept `p` (1D array), `t` (1D array), and then unpacked `args`
        as positional arguments.
    p : numpy.ndarray
        Current parameter values (n-element 1D array).
    t : numpy.ndarray
        Independent variable data (m-element 1D array).
    y_hat : numpy.ndarray
        Model evaluation at current `p`, `t`, i.e., `func(p, t, *args)`.
        This is the output of the model function for the current parameters,
        used as the baseline for finite differencing.
    args : tuple, optional
        Additional positional arguments to pass to the `func` callable.
        Defaults to an empty tuple.
    dp_ratio : float, optional
        Fractional increment base for `p` for numerical derivatives.
        Actual step is `dp_ratio * (1 + abs(p))`. Default is 1e-8.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix (m x n), where m is the length of `y_hat` and n is the
        length of `p`.
    """
    m = y_hat.shape[0]  # m is derived from the output length
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
        y_plus = func(p_temp, t, *args)  # pass *args
        p_temp[j] = p_j_original

        J[:, j] = (y_plus - y_hat) / step
    return J


@jit(nopython=True, cache=True, fastmath=True)
def levenberg_marquardt_core(
    func: ModelFunc,
    t: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
    target_y: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    lambda_up_factor: float = 3.0,
    lambda_down_factor: float = 2.0,
    dp_ratio: float = 1e-8,
    use_marquardt_damping: bool = True,
    jac_func: JacobianFunc | None = None,
    args: tuple = (),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, int, bool]:
    """
    Core Levenberg-Marquardt algorithm for non-linear least squares optimization.

    Implements the standard Levenberg-Marquardt optimization method to find the
    parameters `p` that minimize the sum of squared weighted residuals.
    The objective function is `sum(weights * (target_y - func(p, t, *args))**2)`
    if `target_y` is provided. If `target_y` is None, it minimizes
    `sum(weights * func(p, t, *args)**2)`.

    Supports optional weighting and uses a damping strategy (Marquardt-style
    diagonal scaling or Levenberg-style identity matrix scaling). It can utilize
    an analytical Jacobian function if provided, falling back to a
    Numba-accelerated finite difference calculation otherwise.

    This is the core, single-fit routine designed to be JIT compiled
    for performance and used by higher-level functions like
    :func:`levenberg_marquardt_pixelwise`.

    Parameters
    ----------
    func : callable
        The model function `y_hat = func(p, t, *args)`. It must be a Numba JIT-compilable
        function that accepts a 1D array `p` (parameters), a 1D array `t` (independent variable),
        and then unpacked `args` as additional positional arguments.
        - If `target_y` is provided, `func` should return a 1D array `y_hat`
          of model predictions, with `len(y_hat)` matching `len(target_y)`.
        - If `target_y` is None, `func` should return a 1D array representing
          the residuals directly.
    t : numpy.ndarray
        Independent variable data (m-element 1D array, float64 dtype).
        This is passed as the second argument to `func` and `jac_func`.
    p0 : numpy.ndarray
        Initial guess for the parameters `p` (n-element 1D array, float64 dtype).
        This is passed as the first argument to `func` and `jac_func`.
    target_y : numpy.ndarray or None, optional, default=None
        Dependent variable (experimental) data (m-element 1D array, float64 dtype)
        to be fitted against the model output `func(p, t, *args)`.
        If None, the function `func(p, t, *args)` itself is minimized (i.e., its output
        is treated as the residual to be squared and summed). This is useful
        for direct minimization of a custom objective function.
    weights : numpy.ndarray or None, optional, default=None
        Weights array (m-element 1D array, float64 dtype). The algorithm
        minimizes the sum of weighted squared residuals.
        If None, uniform weights of 1.0 are used. Must be Numba compatible.
        If `target_y` is provided, `len(weights)` must match `len(target_y)`.
        If `target_y` is None, `len(weights)` must match the length of `func(p, *args)` output.
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
        Step size ratio for finite difference Jacobian calculation.
        The step size for parameter `p[j]` is `dp_ratio * (1 + abs(p[j]))`.
    use_marquardt_damping : bool, optional, default=True
        If True, the Marquardt damping scheme is used where `lambda` scales
        the diagonal elements of `J^T W J`. If False, the Levenberg scheme is used
        where `lambda` scales the identity matrix, adding `lambda` to the diagonal.
        Marquardt damping is often preferred as it is scale-invariant with respect
        to parameter scaling.
    jac_func : callable or None, optional, default=None
        Analytical Jacobian function `J = jac_func(p, t, *args)`. It must be a
        Numba JIT-compilable function that accepts a 1D array `p`, a 1D array `t`,
        and then unpacked `args` as additional positional arguments. It should return
        the (m x n) Jacobian matrix as a 2D NumPy array (float64 dtype),
        where `m` is the number of residuals/model points and `n` is the
        number of parameters. If None, the Jacobian is computed using
        finite differences via :func:`_lm_finite_difference_jacobian`.
    args : tuple, optional
        Additional positional arguments to pass to the `func` and `jac_func` callables.
        Defaults to an empty tuple. These arguments are passed as `*args`.

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
    for Numba acceleration. Ensure that `func`, `jac_func` (if provided), and
    all operations within them are Numba-compatible.

    Uses a small epsilon value (`_EPS`) derived from machine precision for
    numerical stability, e.g., in checks for near-zero denominators or matrix
    diagonals.

    Convergence is checked *after* a successful step (where Chi-squared decreases).

    The covariance matrix `(J^T W J)^-1` is a standard output of LM. If weights `W`
    represent `1/sigma_i^2` (inverse variances), this is the parameter covariance.
    If weights are uniform (unweighted least squares), this needs to be scaled
    by the reduced Chi-squared (`chi2 / (m - n)`) to estimate the parameter
    covariance matrix assuming independent, identically distributed errors. This
    scaling is **not** performed by this core function; it returns `(J^T W J)^-1` directly.

    For a detailed description of the algorithm and its convergence properties,
    see :cite:p:`levenberg1944method`, :cite:p:`marquardt1963algorithm`,
    and :cite:p:`nocedal2006numerical`. The implementation loosely follows
    concepts described by :cite:p:`gavin2020levenberg`.

    References
    ----------
    .. bibliography:: refs.bib
       :filter: docname in ("__init__")
       :style: plain

    See Also
    --------
    :func:`levenberg_marquardt_pixelwise` : Applies this core algorithm to 3D pixel data.
    :func:`_lm_finite_difference_jacobian` : The internal function used for FD Jacobian when `jac_func` is None.

    """
    n = p0.shape[0]
    p = p0.copy()

    # Determine m (number of residuals/measurements) and initial residuals
    y_hat = func(p, t, *args)  # Call func with p, t, and *args
    m = y_hat.shape[0]  # m is length of the function output

    if target_y is None:
        # If target_y is None, func's output is treated as the residuals
        residuals_are_func_output = True
        res = y_hat
        if weights is None:
            W_arr = np.ones(m, dtype=p0.dtype)
        else:
            if weights.shape[0] != m:
                raise ValueError(
                    f"Length of `weights` ({weights.shape[0]}) must match length of `func` output ({m}) when `target_y` is None."
                )
            W_arr = weights.copy()
    else:
        # If target_y is provided, residuals = target_y - func(p, t, *args)
        if target_y.shape[0] != m:
            raise ValueError(
                f"Length of `target_y` ({target_y.shape[0]}) must match length of `func` output ({m})."
            )

        residuals_are_func_output = False
        res = target_y - y_hat
        if weights is None:
            W_arr = np.ones(m, dtype=target_y.dtype)
        else:
            if weights.shape[0] != m:
                raise ValueError(
                    f"Length of `weights` ({weights.shape[0]}) must match length of `target_y` ({m})."
                )
            W_arr = weights.copy()

    # Calculate initial Chi-squared and JtWres
    W_res = W_arr * res
    J: npt.NDArray[np.float64]
    if jac_func is not None:
        J = jac_func(p, t, *args)  # Pass *args
    else:
        J = _lm_finite_difference_jacobian(
            func, p, t, y_hat, args, dp_ratio
        )  # Pass func, p, t, y_hat, args

    if W_arr.ndim == 1:
        W_J = W_arr[:, np.newaxis] * J  # Apply weights to Jacobian
    else:
        # This branch would be for a 2D weight matrix, which is not currently supported by weights_1d
        # but theoretically possible if W_arr was 2D. Let's keep the existing logic.
        W_J = W_arr * J

    JtWJ = J.T @ W_J
    JtWres = J.T @ W_res  # J^T W (residuals) - this is the effective gradient
    chi2 = np.sum(W_res**2)  # Chi-squared is sum of weighted squared residuals

    current_max_grad = np.max(np.abs(JtWres))  # Check initial gradient
    if current_max_grad < tol_g:
        converged = True
        n_iter_final = 0
        final_cov_p = np.full((n, n), np.nan, dtype=p.dtype)
        try:
            if (m - n > 0) and np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ)
            else:
                final_cov_p = np.full((n, n), np.inf, dtype=p.dtype)
        except Exception:
            final_cov_p = np.full((n, n), np.inf, dtype=p.dtype)
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

        A: npt.NDArray[np.float64]
        if use_marquardt_damping:
            diag_JtWJ = np.diag(JtWJ)
            diag_JtWJ_stable = diag_JtWJ + _EPS * (diag_JtWJ == 0.0)
            A = JtWJ + lambda_val * np.diag(diag_JtWJ_stable)
        else:
            A = JtWJ + lambda_val * np.eye(n, dtype=JtWJ.dtype)

        dp_step: npt.NDArray[np.float64]
        try:
            dp_step = np.linalg.solve(A, JtWres)  # Solve with JtWres (gradient)
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
        y_hat_try = func(p_try, t, *args)  # Pass *args

        if not np.all(np.isfinite(y_hat_try)):
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

        # Calculate residuals for trial step
        if residuals_are_func_output:
            res_try = y_hat_try
        else:
            res_try = target_y - y_hat_try

        W_res_try = W_arr * res_try
        chi2_try = np.sum(W_res_try**2)

        if chi2_try < chi2_at_iter_start:
            lambda_val /= lambda_down_factor
            lambda_val = np.maximum(lambda_val, 1e-15)

            p = p_try
            chi2 = chi2_try
            y_hat = y_hat_try  # Update y_hat for next iteration's calculations

            if jac_func is not None:
                J = jac_func(p, t, *args)  # Pass *args
            else:
                J = _lm_finite_difference_jacobian(
                    func, p, t, y_hat, args, dp_ratio
                )  # Pass func, p, t, y_hat, args

            if W_arr.ndim == 1:
                W_J = W_arr[:, np.newaxis] * J
            else:
                W_J = W_arr * J
            JtWJ = J.T @ W_J
            # Recalculate JtWres with updated p and y_hat
            if residuals_are_func_output:
                res = y_hat
            else:
                res = target_y - y_hat
            JtWres = J.T @ (W_arr * res)

            current_max_grad_after_step = np.max(np.abs(JtWres))
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

        else:
            lambda_val *= lambda_up_factor
            lambda_val = np.minimum(lambda_val, 1e15)
            if lambda_val > 1e12:
                converged = False
                break
            continue

    if not converged and n_iter_final == max_iter:
        current_max_grad_at_max_iter = np.max(np.abs(JtWres))
        if current_max_grad_at_max_iter < tol_g:
            converged = True

    final_cov_p = np.full((n, n), np.inf, dtype=p.dtype)
    try:
        dof = m - n
        if dof > 0:
            if np.any(np.abs(JtWJ) > _EPS):
                final_cov_p = np.linalg.inv(JtWJ)
        # else: leave as np.inf (dof <= 0)
    except Exception:
        final_cov_p = np.full((n, n), np.inf, dtype=p.dtype)

    return p, final_cov_p, chi2, n_iter_final, converged


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def levenberg_marquardt_pixelwise(
    func: ModelFunc,
    t_common: npt.NDArray[np.float64],  # Independent variable 't' is now explicit here
    p0_global: npt.NDArray[np.float64],
    target_y_3d: npt.NDArray[np.float64] | None = None,
    weights_1d: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    lambda_up_factor: float = 3.0,
    lambda_down_factor: float = 2.0,
    dp_ratio: float = 1e-8,
    use_marquardt_damping: bool = True,
    jac_func: JacobianFunc | None = None,
    args_for_each_pixel: tuple = (),
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
]:
    """
    Applies Levenberg-Marquardt fitting pixel-wise to 3D data.

    Numba JIT compiled with `parallel=True` for performing fits on multiple
    pixels concurrently using `numba.prange`. Each pixel's curve
    is fitted independently using :func:`levenberg_marquardt_core`.

    Parameters
    ----------
    func : callable
        Model function `y_hat = func(p, t, *args)`. It must be a Numba JIT-compilable
        function that accepts a 1D array `p` (parameters), a 1D array `t` (independent variable),
        and then unpacked `args` as additional positional arguments.
        - If `target_y_3d` is provided, `func` should return a 1D array `y_hat`
          of model predictions. The length must match `t_common.shape[0]`.
        - If `target_y_3d` is None, `func` should return a 1D array representing
          the residuals directly. The length must match `t_common.shape[0]`.
    t_common : numpy.ndarray
        Independent variable data, common for all pixels (m-element 1D array).
        This is passed as the `t` argument to `func` and `jac_func` for each pixel.
    p0_global : numpy.ndarray
        Global initial guess for parameters (n-element 1D array),
        used for each pixel's fit.
    target_y_3d : numpy.ndarray or None, optional, default=None
        Dependent variable data (rows x cols x m_depth_points 3D array).
        Each `target_y_3d[r, c, :]` is a curve to be fitted.
        If None, the `func` itself is minimized (its output treated as residual).
        `target_y_3d.shape[2]` must match `t_common.shape[0]`.
    weights_1d : numpy.ndarray or None, optional
        1D array of weights (m-element), applied identically to each pixel's fit.
        If None (default), uniform weights (1.0) are used. Must be Numba compatible.
        Its length `m` must match `t_common.shape[0]`.
    max_iter, tol_g, ..., use_marquardt_damping :
        Parameters for :func:`levenberg_marquardt_core` (see its docstring).
    dp_ratio : float, optional
        Step size ratio for finite difference Jacobian calculation.
        Default is 1e-8.
    jac_func : callable or None, optional
        Analytical Jacobian function `J = jac_func(p, t, *args)`. Must be Numba
        JIT-compilable and accept a 1D array `p`, a 1D array `t`, and then
        unpacked `args` as additional positional arguments. It should return
        the (m x n) Jacobian as a 2D NumPy array. If None (default), the finite
        difference Jacobian is calculated internally.
    args_for_each_pixel : tuple, optional
        Additional positional arguments to pass to the `func` and `jac_func` callables
        for *each* pixel. Defaults to an empty tuple. These arguments are passed as `*args`.

    Returns
    -------
    tuple
        Contains 3D/2D arrays corresponding to the outputs of
        :func:`levenberg_marquardt_core` for each pixel:
        - p_results : numpy.ndarray (rows x cols x n_params)
            Fitted parameters for each pixel. Contains `np.nan` for skipped pixels.
        - cov_p_results : numpy.ndarray (rows x cols x n_params x n_params)
            Covariance matrices for each pixel. Contains `np.inf`/`np.nan` for
            non-calculable/skipped pixels.
        - chi2_results : numpy.ndarray (rows x cols)
            Final weighted Chi-squared values for each pixel. Contains `np.nan`
            for skipped pixels.
        - n_iter_results : numpy.ndarray (rows x cols, dtype=int)
            Number of iterations for each pixel. Contains 0 for skipped pixels.
        - converged_results : numpy.ndarray (rows x cols, dtype=bool)
            Convergence status for each pixel. Contains False for skipped pixels.
    """
    rows: int
    cols: int
    # num_measurements_per_pixel: int # `m` from core LM - now implicit from t_common

    # The number of measurements per pixel is always given by t_common
    num_measurements_per_pixel = t_common.shape[0]

    if target_y_3d is not None:
        if target_y_3d.shape[2] != num_measurements_per_pixel:
            raise ValueError(
                f"Third dimension of `target_y_3d` ({target_y_3d.shape[2]}) must match length of `t_common` ({num_measurements_per_pixel})."
            )
        rows, cols, _ = target_y_3d.shape
    else:
        # If target_y_3d is None, we need rows/cols explicitly, as they cannot be inferred
        # from t_common or p0_global's global nature.
        # This function is primarily designed for image data, so target_y_3d is usually present.
        raise ValueError(
            "`target_y_3d` must be provided for pixel-wise fitting "
            "to define the image dimensions (rows, cols)."
        )

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

        # Extract pixel-specific target_y data
        pixel_target_y = target_y_3d[r, c, :]

        # Skip pixels with NaN data (e.g., masked pixels)
        if np.any(np.isnan(pixel_target_y)):
            continue

        p0_pixel = p0_global.copy()

        p_fit, cov_p, chi2_val, iters, conv_flag = levenberg_marquardt_core(
            func,
            t_common,  # Pass t_common explicitly to core LM
            p0_pixel,
            target_y=pixel_target_y,  # Pass pixel-specific target_y
            weights=weights_1d,  # Pass the 1D weights array (same for all pixels)
            max_iter=max_iter,
            tol_g=tol_g,
            tol_p=tol_p,
            tol_c=tol_c,
            lambda_0_factor=lambda_0_factor,
            lambda_up_factor=lambda_up_factor,
            lambda_down_factor=lambda_down_factor,
            dp_ratio=dp_ratio,
            use_marquardt_damping=use_marquardt_damping,
            jac_func=jac_func,
            args=args_for_each_pixel,  # Pass args
        )

        p_results[r, c, :] = p_fit
        cov_p_results[r, c, :, :] = cov_p
        chi2_results[r, c] = chi2_val
        n_iter_results[r, c] = iters
        converged_results[r, c] = conv_flag
    return p_results, cov_p_results, chi2_results, n_iter_results, converged_results
