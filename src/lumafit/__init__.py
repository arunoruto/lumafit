"""
Lumafit: A Numba-accelerated Levenberg-Marquardt Fitting Library.

This library provides functions for performing non-linear least squares
fitting using the Levenberg-Marquardt algorithm
:cite:p:`levenberg1944method,marquardt1963algorithm`, with emphasis on
performance acceleration via Numba :cite:p:`lam2015numba` JIT compilation
and parallel execution.
It is particularly suited for fitting large numbers of curves, such as
pixel-wise fitting in 3D image data.

`fit_curve` fits one curve and `fit_cube` every pixel of an image cube,
both for models that return a whole curve. `fit_cube_pointwise` fits a
cube with a model that computes one sample at a time, on the CPU or on a
GPU.
"""

from collections.abc import Callable
from importlib.metadata import version

import numpy as np
import numpy.typing as npt
from numba import jit, prange

from lumafit._result import FitResult
from lumafit.pointwise import fit_cube_pointwise

__all__ = ["FitResult", "fit_cube", "fit_cube_pointwise", "fit_curve"]
__version__ = version(__package__) if __package__ else "0.0.0"

_EPS = np.finfo(float).eps
_LAMBDA_MIN = 1e-15
_LAMBDA_MAX = 1e12
# Everything `fastmath=True` enables except `nnan` and `ninf`: masked (NaN)
# pixels and non-finite model output have to stay detectable.
_FASTMATH = {"reassoc", "nsz", "arcp", "contract", "afn"}

# Model:    model(p, *args) -> y_hat (m,), or the residuals if no data is given
# Jacobian: jac(p, *args) -> J (m, n)
# Both must be Numba nopython functions.
ModelFunc = Callable[..., npt.NDArray[np.float64]]
JacobianFunc = Callable[..., npt.NDArray[np.float64]]


@jit(nopython=True, cache=True, fastmath=_FASTMATH)
def _finite_difference_jacobian(
    model: ModelFunc,
    p: npt.NDArray[np.float64],
    y_hat: npt.NDArray[np.float64],
    args: tuple = (),
    dp_ratio: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """
    Approximate the model Jacobian with forward differences.

    Costs one model evaluation per parameter. The default step is close
    to the square root of the machine precision, which balances
    truncation against rounding error :cite:p:`nocedal2006numerical`.

    Parameters
    ----------
    model : callable
        Model function ``model(p, *args)``.
    p : numpy.ndarray
        Parameters at which to differentiate, shape ``(n,)``.
    y_hat : numpy.ndarray
        Model output at `p`, shape ``(m,)``.
    args : tuple, optional
        Extra arguments passed to `model`.
    dp_ratio : float, default 1e-8
        Relative step; parameter ``j`` is perturbed by
        ``dp_ratio * (1 + abs(p[j]))``.

    Returns
    -------
    numpy.ndarray
        Jacobian ``d y_hat / d p``, shape ``(m, n)``.
    """
    n = p.shape[0]
    J = np.empty((y_hat.shape[0], n), dtype=p.dtype)
    p_step = p.copy()
    for j in range(n):
        step = dp_ratio * (1.0 + abs(p[j]))
        p_step[j] = p[j] + step
        J[:, j] = (model(p_step, *args) - y_hat) / step
        p_step[j] = p[j]
    return J


@jit(nopython=True, cache=True, fastmath=_FASTMATH)
def _normal_equations(model, jac, p, y_hat, res, w, args, dp_ratio):
    """Return ``J^T W J`` and the gradient ``J^T W res`` at `p`."""
    if jac is None:
        J = _finite_difference_jacobian(model, p, y_hat, args, dp_ratio)
    else:
        J = jac(p, *args)
    WJ = w[:, np.newaxis] * J
    return J.T @ WJ, WJ.T @ res


@jit(nopython=True, cache=True, fastmath=_FASTMATH)
def fit_curve(
    model: ModelFunc,
    p0: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    dp_ratio: float = 1e-8,
    use_marquardt_damping: bool = True,
    jac: JacobianFunc | None = None,
    args: tuple = (),
) -> FitResult:
    r"""
    Fit a single curve with the Levenberg-Marquardt algorithm.

    Minimizes the weighted sum of squared residuals
    ``sum(weights * (y - model(p, *args))**2)``. Being a Numba nopython
    function, it can also be called from other compiled code.

    Parameters
    ----------
    model : callable
        Model ``y_hat = model(p, *args)``, a Numba nopython function
        that returns a 1D float64 array, e.g. ``def model(p, t)`` for
        ``args=(t,)``. If `y` is None, its output is taken as the
        residuals.
    p0 : numpy.ndarray
        Initial guess for the parameters, shape ``(n,)``.
    y : numpy.ndarray, optional
        Data to fit, shape ``(m,)``. If None, the output of `model`
        itself is minimized.
    weights : numpy.ndarray, optional
        Weights, shape ``(m,)``; typically ``1 / sigma**2``. Uniform
        weights are used if None.
    max_iter : int, default 100
        Maximum number of iterations (trial steps).
    tol_g : float, default 1e-8
        Converged when the largest absolute component of the gradient
        ``J^T W res`` is below this value.
    tol_p : float, default 1e-8
        Converged when the largest relative parameter change of an
        accepted step is below this value.
    tol_c : float, default 1e-8
        Converged when the relative decrease of chi-squared of an
        accepted step is below this value.
    lambda_0_factor : float, default 1e-2
        Initial damping. Used as is with Marquardt damping and scaled by
        the largest diagonal element of ``J^T W J`` otherwise.
    dp_ratio : float, default 1e-8
        Relative step of the finite difference Jacobian; parameter ``j``
        is perturbed by ``dp_ratio * (1 + abs(p[j]))``. Unused if `jac`
        is given.
    use_marquardt_damping : bool, default True
        Damp with ``lambda * diag(J^T W J)``, which makes the step
        invariant to the scale of the parameters
        :cite:p:`marquardt1963algorithm`. If False, damp with
        ``lambda * I`` :cite:p:`levenberg1944method`.
    jac : callable, optional
        Analytical Jacobian ``J = jac(p, *args)`` returning a float64
        array of shape ``(m, n)``; a Numba nopython function. Forward
        differences are used if None.
    args : tuple, optional
        Extra arguments passed to `model` and `jac`, e.g. the
        independent variable.

    Returns
    -------
    FitResult
        The named tuple ``(p, cov, chi2, n_iter, converged)``, see
        `FitResult`.

    Raises
    ------
    ValueError
        If the length of `y` or `weights` does not match the output of
        `model`.

    See Also
    --------
    fit_cube : Fit every pixel of an image cube.

    Notes
    -----
    The damping parameter follows the gain-ratio update of
    :cite:t:`nielsen1999damping`: a step is accepted if it lowers
    chi-squared, the damping then shrinks by up to a factor of 3
    depending on how well the quadratic model predicted the decrease,
    and it grows by 2, 4, 8, ... on consecutive rejected steps. For
    background see :cite:t:`nocedal2006numerical` and
    :cite:t:`gavin2020levenberg`; :cite:t:`more1978levenbergmarquardt`
    describes the trust-region variant implemented in MINPACK.

    ``cov`` is the parameter covariance if `weights` are inverse
    variances. For an unweighted fit, multiply it by the reduced
    chi-squared ``chi2 / (m - n)`` :cite:p:`gavin2020levenberg`. When
    the fit stops on `tol_p` or `tol_c`, ``cov`` is evaluated one (tiny)
    step before ``p``, which saves a Jacobian evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from lumafit import fit_curve
    >>> @njit
    ... def decay(p, t):
    ...     return p[0] * np.exp(-t / p[1])
    >>> t = np.linspace(0.0, 5.0, 50)
    >>> y = decay(np.array([3.0, 1.5]), t)
    >>> result = fit_curve(decay, np.array([1.0, 1.0]), y, args=(t,))
    >>> np.round(result.p, 6), result.converged
    (array([3. , 1.5]), True)
    """
    n = p0.shape[0]
    p = p0.astype(np.float64)
    y_hat = model(p, *args)
    m = y_hat.shape[0]

    if weights is None:
        w = np.ones(m, dtype=y_hat.dtype)
    else:
        if weights.shape[0] != m:
            raise ValueError(
                f"Length of `weights` ({weights.shape[0]}) must match length of `model` output ({m})."
            )
        w = np.asarray(weights, np.float64)
    if y is None:
        target = np.zeros(m, dtype=np.float64)
    else:
        if y.shape[0] != m:
            raise ValueError(
                f"Length of `y` ({y.shape[0]}) must match length of `model` output ({m})."
            )
        target = np.asarray(y, np.float64)

    res = target - y_hat
    chi2 = np.sum(w * res * res)
    JtWJ, g = _normal_equations(model, jac, p, y_hat, res, w, args, dp_ratio)

    lam = lambda_0_factor
    if not use_marquardt_damping:
        lam *= np.max(np.diag(JtWJ))
    nu = 2.0  # growth factor of `lam` for the next rejected step

    n_iter = 0
    converged = np.max(np.abs(g)) < tol_g
    while not converged and n_iter < max_iter:
        n_iter += 1

        if use_marquardt_damping:
            D = np.maximum(np.diag(JtWJ), _EPS)
        else:
            D = np.ones(n, dtype=p.dtype)

        try:
            dp = np.linalg.solve(JtWJ + lam * np.diag(D), g)
        except Exception:  # singular matrix
            dp = np.full(n, np.nan, dtype=p.dtype)

        accepted = False
        rho = 0.0  # actual / predicted decrease of chi2
        rel_chi2_decrease = 1.0
        if np.all(np.isfinite(dp)):
            p_try = p + dp
            y_try = model(p_try, *args)
            res_try = target - y_try
            chi2_try = np.sum(w * res_try * res_try)
            # Decrease of chi2 predicted by the damped quadratic model
            predicted = np.sum(dp * (lam * D * dp + g))
            if chi2_try < chi2 and predicted > 0.0:  # False for NaN
                accepted = True
                rho = (chi2 - chi2_try) / predicted
                if chi2 > _EPS:
                    rel_chi2_decrease = (chi2 - chi2_try) / chi2
                p, y_hat, res, chi2 = p_try, y_try, res_try, chi2_try

        if not accepted:
            lam *= nu
            nu *= 2.0
            if lam > _LAMBDA_MAX:
                break
            continue

        lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), _LAMBDA_MIN)
        nu = 2.0

        if (n_iter > 1 and rel_chi2_decrease < tol_c) or np.max(
            np.abs(dp) / (np.abs(p) + _EPS)
        ) < tol_p:
            converged = True
            break

        JtWJ, g = _normal_equations(model, jac, p, y_hat, res, w, args, dp_ratio)
        converged = np.max(np.abs(g)) < tol_g

    cov_p = np.full((n, n), np.inf, dtype=p.dtype)
    if m > n:
        try:
            cov_p = np.linalg.inv(JtWJ)
        except Exception:  # singular matrix
            cov_p = np.full((n, n), np.inf, dtype=p.dtype)

    return FitResult(p, cov_p, chi2, n_iter, converged)


@jit(nopython=True, parallel=True, cache=True, fastmath=_FASTMATH)
def fit_cube(
    model: ModelFunc,
    p0: npt.NDArray[np.float64],
    cube: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol_g: float = 1e-8,
    tol_p: float = 1e-8,
    tol_c: float = 1e-8,
    lambda_0_factor: float = 1e-2,
    dp_ratio: float = 1e-8,
    use_marquardt_damping: bool = True,
    jac: JacobianFunc | None = None,
    args: tuple = (),
) -> FitResult:
    """
    Fit every pixel of an image cube in parallel.

    Each curve ``cube[r, c, :]`` is fitted independently with
    `fit_curve`, spread over all Numba threads.

    Parameters
    ----------
    model : callable
        Model ``y_hat = model(p, *args)``, see `fit_curve`.
    p0 : numpy.ndarray
        Initial guess used for every pixel, shape ``(n,)``.
    cube : numpy.ndarray
        Data to fit, shape ``(rows, cols, m)``. Pixels that contain a
        NaN are skipped.
    weights : numpy.ndarray, optional
        Weights shared by all pixels, shape ``(m,)``.
    max_iter, tol_g, tol_p, tol_c, lambda_0_factor, dp_ratio
        See `fit_curve`.
    use_marquardt_damping, jac
        See `fit_curve`.
    args : tuple, optional
        Extra arguments passed to `model` and `jac`; the same tuple is
        used for every pixel.

    Returns
    -------
    FitResult
        The named tuple ``(p, cov, chi2, n_iter, converged)`` with
        shapes ``(rows, cols, n)``, ``(rows, cols, n, n)`` and three
        times ``(rows, cols)``, see `FitResult`.

    See Also
    --------
    fit_curve : The underlying single-curve fit.
    fit_cube_pointwise : The same fit on a GPU, for pointwise models.

    Notes
    -----
    The first call compiles the solver for the given `model` and takes
    around ten seconds; later calls reuse the compiled code. The number
    of threads is controlled with ``numba.set_num_threads``.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from lumafit import fit_cube
    >>> @njit
    ... def decay(p, t):
    ...     return p[0] * np.exp(-t / p[1])
    >>> t = np.linspace(0.0, 5.0, 50)
    >>> cube = np.tile(decay(np.array([3.0, 1.5]), t), (4, 4, 1))
    >>> result = fit_cube(decay, np.array([1.0, 1.0]), cube, args=(t,))
    >>> result.p.shape, bool(result.converged.all())
    ((4, 4, 2), True)
    """
    rows, cols, _ = cube.shape
    n = p0.shape[0]

    p_results = np.full((rows, cols, n), np.nan, dtype=np.float64)
    cov_p_results = np.full((rows, cols, n, n), np.nan, dtype=np.float64)
    chi2_results = np.full((rows, cols), np.nan, dtype=np.float64)
    n_iter_results = np.zeros((rows, cols), dtype=np.int64)
    converged_results = np.zeros((rows, cols), dtype=np.bool_)

    for flat_idx in prange(rows * cols):
        r = flat_idx // cols
        c = flat_idx % cols
        pixel_y = cube[r, c, :]
        if np.any(np.isnan(pixel_y)):  # masked pixel
            continue

        p_fit, cov_p, chi2, n_iter, converged = fit_curve(
            model,
            p0,
            y=pixel_y,
            weights=weights,
            max_iter=max_iter,
            tol_g=tol_g,
            tol_p=tol_p,
            tol_c=tol_c,
            lambda_0_factor=lambda_0_factor,
            dp_ratio=dp_ratio,
            use_marquardt_damping=use_marquardt_damping,
            jac=jac,
            args=args,
        )

        p_results[r, c, :] = p_fit
        cov_p_results[r, c, :, :] = cov_p
        chi2_results[r, c] = chi2
        n_iter_results[r, c] = n_iter
        converged_results[r, c] = converged
    return FitResult(
        p_results, cov_p_results, chi2_results, n_iter_results, converged_results
    )
