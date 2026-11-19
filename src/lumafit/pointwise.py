"""
Pixel-wise fitting with a pointwise model, on the CPU or a Vulkan GPU.

The solver in this module evaluates the model one sample at a time and
allocates no arrays of run-time size. That makes the same code compile for
CPU threads (:cite:p:`lam2015numba`) and, through numba-vulkan, for any GPU
with a Vulkan driver.
"""

import functools
import math

import numpy as np
from numba import njit, prange

from lumafit._result import FitResult

TARGETS = ("cpu", "vulkan")


def _make_core(model, n, dtype):
    """Compile the single-curve solver for `n` parameters of type `dtype`.

    Every constant is typed, so a float32 solver never computes in float64.
    """
    F = dtype
    ZERO, ONE, TWO, THIRD = F(0), F(1), F(2), F(1 / 3)
    EPS, INF, NAN = F(np.finfo(dtype).eps), F(np.inf), F(np.nan)
    LAMBDA_MIN, LAMBDA_MAX = F(1e-15), F(1e12)

    @njit
    def normal_equations(p, x, y, w, dp_ratio, A, g, pt, row):
        """Fill the lower triangle of ``J^T W J`` and ``J^T W r``; return chi2."""
        for a in range(n):
            g[a] = ZERO
            pt[a] = p[a]
            for b in range(a + 1):
                A[a, b] = ZERO
        chi2 = ZERO
        for k in range(y.shape[0]):
            f0 = model(p, x[k])
            r = y[k] - f0
            chi2 += w[k] * r * r
            for a in range(n):  # forward differences
                h = dp_ratio * (ONE + abs(p[a]))
                pt[a] = p[a] + h
                row[a] = (model(pt, x[k]) - f0) / h
                pt[a] = p[a]
            for a in range(n):
                g[a] += w[k] * row[a] * r
                for b in range(a + 1):
                    A[a, b] += w[k] * row[a] * row[b]
        return chi2

    @njit
    def solve(A, lam, b, L, out):
        """Solve ``(A + lam * diag(A)) out = b`` by Cholesky decomposition.

        Returns False if the matrix is not positive definite.
        """
        for j in range(n):
            s = A[j, j] + lam * max(A[j, j], EPS)
            for k in range(j):
                s -= L[j, k] * L[j, k]
            if not s > ZERO:  # also False for NaN
                return False
            L[j, j] = math.sqrt(s)
            for i in range(j + 1, n):
                s = A[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                L[i, j] = s / L[j, j]
        for i in range(n):  # L z = b
            s = b[i]
            for k in range(i):
                s -= L[i, k] * out[k]
            out[i] = s / L[i, i]
        for i in range(n - 1, -1, -1):  # L^T out = z
            s = out[i]
            for k in range(i + 1, n):
                s -= L[k, i] * out[k]
            out[i] = s / L[i, i]
        return True

    @njit
    def max_abs(v):
        m = ZERO
        for a in range(n):
            m = max(m, abs(v[a]))
        return m

    @njit
    def fit_one(y, x, w, p, cov, max_iter, tol_g, tol_p, tol_c, lam, dp_ratio):
        """Fit one curve in place of `p`; the algorithm of `lumafit.fit_curve`."""
        A = np.empty((n, n), F)
        L = np.empty((n, n), F)
        g = np.empty(n, F)
        dp = np.empty(n, F)
        pt = np.empty(n, F)
        row = np.empty(n, F)

        for k in range(y.shape[0]):
            if math.isnan(y[k]):  # masked pixel
                for a in range(n):
                    p[a] = NAN
                    for b in range(n):
                        cov[a, b] = NAN
                return NAN, 0, False

        chi2 = normal_equations(p, x, y, w, dp_ratio, A, g, pt, row)
        nu = TWO
        n_iter = 0
        converged = max_abs(g) < tol_g
        while (not converged) and n_iter < max_iter:
            n_iter += 1
            accepted = False
            rho = ZERO
            rel_decrease = ONE
            chi2_try = chi2
            if solve(A, lam, g, L, dp):
                predicted = ZERO
                for a in range(n):
                    pt[a] = p[a] + dp[a]
                    predicted += dp[a] * (lam * max(A[a, a], EPS) * dp[a] + g[a])
                chi2_try = ZERO
                for k in range(y.shape[0]):
                    r = y[k] - model(pt, x[k])
                    chi2_try += w[k] * r * r
                if chi2_try < chi2 and predicted > ZERO:  # False for NaN
                    accepted = True
                    rho = (chi2 - chi2_try) / predicted
                    if chi2 > EPS:
                        rel_decrease = (chi2 - chi2_try) / chi2
            if not accepted:
                lam *= nu
                nu *= TWO
                if lam > LAMBDA_MAX:
                    break
                continue

            step = ZERO
            for a in range(n):
                p[a] = pt[a]
                step = max(step, abs(dp[a]) / (abs(pt[a]) + EPS))
            q = TWO * rho - ONE
            lam = max(lam * max(THIRD, ONE - q * q * q), LAMBDA_MIN)
            nu = TWO
            if (n_iter > 1 and rel_decrease < tol_c) or step < tol_p:
                chi2 = chi2_try
                converged = True
                break
            chi2 = normal_equations(p, x, y, w, dp_ratio, A, g, pt, row)
            converged = max_abs(g) < tol_g

        # Covariance (J^T W J)^-1, column by column
        invertible = y.shape[0] > n
        for b in range(n):
            if invertible:
                for a in range(n):
                    g[a] = ONE if a == b else ZERO
                invertible = solve(A, ZERO, g, L, dp)
            for a in range(n):
                cov[a, b] = dp[a] if invertible else INF
        if not invertible:
            for a in range(n):
                for b in range(n):
                    cov[a, b] = INF
        return chi2, n_iter, converged

    return fit_one


@functools.lru_cache(maxsize=None)
def _compile(model, n, dtype, target):
    """Compile the cube fit for one model, size, precision and target."""
    fit_one = _make_core(model, n, dtype)

    if target == "cpu":

        @njit(parallel=True)
        def run(Y, x, w, p0, P, cov, chi2, n_iter, conv, max_iter, tg, tp, tc, lam, h):
            for i in prange(Y.shape[0]):
                for a in range(n):
                    P[i, a] = p0[a]
                chi2[i], n_iter[i], conv[i] = fit_one(
                    Y[i], x, w, P[i], cov[i], max_iter, tg, tp, tc, lam, h
                )

        def launch(device, inputs, outputs, scalars):
            run(*inputs, *outputs, *scalars)
            return outputs

        return launch

    import numba_vulkan as nv

    @nv.jit
    def kernel(Y, x, w, p0, P, cov, chi2, n_iter, conv, max_iter, tg, tp, tc, lam, h):
        i = nv.global_id(0)
        if i < Y.shape[0]:
            for a in range(n):
                P[i, a] = p0[a]
            chi2[i], n_iter[i], conv[i] = fit_one(
                Y[i], x, w, P[i], cov[i], max_iter, tg, tp, tc, lam, h
            )

    def launch(device, inputs, outputs, scalars):
        # Explicit device arrays: handing NumPy arrays to the kernel makes
        # numba-vulkan upload the outputs too and is several times slower.
        on_device = [nv.device_array(o.shape, o.dtype, device=device) for o in outputs]
        kernel.forall(inputs[0].shape[0], device=device)(
            *(nv.to_device(i, device=device) for i in inputs), *on_device, *scalars
        )
        return [d.copy_to_host(o) for d, o in zip(on_device, outputs, strict=True)]

    return launch


def fit_cube_pointwise(
    model,
    p0,
    cube,
    x,
    weights=None,
    max_iter=100,
    tol_g=None,
    tol_p=None,
    tol_c=None,
    lambda_0_factor=1e-2,
    dp_ratio=None,
    *,
    target="cpu",
    device=None,
    dtype=None,
):
    """
    Fit every pixel of an image cube with a pointwise model.

    Unlike `lumafit.fit_cube`, the model computes one sample at a time.
    In exchange the fit can run on a GPU, compiles in a second or two
    and is a little faster on the CPU as well.

    Parameters
    ----------
    model : callable
        ``model(p, x) -> float``, a function compiled with
        ``numba.njit`` that returns the model value for the parameters
        `p` at one sample ``x`` of the independent variable.
    p0 : array_like
        Initial guess used for every pixel, shape ``(n,)``.
    cube : numpy.ndarray
        Data to fit, shape ``(rows, cols, m)``. Pixels that contain a
        NaN are skipped.
    x : array_like
        Independent variable, with one entry per sample along its first
        axis; `model` receives ``x[k]``.
    weights : array_like, optional
        Weights shared by all pixels, shape ``(m,)``.
    max_iter : int, default 100
        Maximum number of iterations per pixel.
    tol_g, tol_p, tol_c : float, optional
        Convergence tolerances, see `lumafit.fit_curve`. They default
        to the square root of the machine precision of `dtype`.
    lambda_0_factor : float, default 1e-2
        Initial damping.
    dp_ratio : float, optional
        Relative step of the finite difference Jacobian. Defaults to
        the square root of the machine precision of `dtype`.
    target : {'cpu', 'vulkan'}, optional
        Run on all Numba threads, or on a Vulkan device through
        numba-vulkan (``pip install lumafit[vulkan]``).
    device : int or str, optional
        Vulkan device by index or part of its name; the first discrete
        GPU by default. See ``numba_vulkan.list_devices``.
    dtype : {numpy.float64, numpy.float32}, optional
        Precision of the computation. Defaults to float32 for a
        float32 `cube` and to float64 otherwise. Most GPUs are several
        times faster in float32, where results agree with float64 to
        about 1e-4.

    Returns
    -------
    lumafit.FitResult
        The named tuple ``(p, cov, chi2, n_iter, converged)`` with the
        shapes of `lumafit.fit_cube`, in `dtype`; ``n_iter`` is int32.

    Raises
    ------
    ValueError
        If `target` or `dtype` is not supported, or the shapes of `p0`,
        `x` or `weights` do not fit `cube`.

    See Also
    --------
    lumafit.fit_cube : The same fit for models that return whole curves.

    Notes
    -----
    The algorithm is that of `lumafit.fit_curve` with Marquardt damping
    and a forward difference Jacobian; an analytical Jacobian is not
    supported. The solver is compiled on the first call for a given
    `model`, number of parameters, `dtype` and `target`, and reused
    afterwards.

    In a float32 fit every operation of the model has to stay in
    float32: a Python literal such as ``0.5 * p[0]`` promotes the
    expression to float64, which numba-vulkan reports with a warning.
    Use typed constants (``HALF = np.float32(0.5)``) instead.

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from numba import njit
    >>> from lumafit import fit_cube_pointwise
    >>> @njit
    ... def decay(p, t):
    ...     return p[0] * math.exp(-t / p[1])
    >>> t = np.linspace(0.0, 5.0, 50)
    >>> cube = np.tile(3.0 * np.exp(-t / 1.5), (4, 4, 1))
    >>> result = fit_cube_pointwise(decay, [1.0, 1.0], cube, t)
    >>> np.round(result.p[0, 0], 6), bool(result.converged.all())
    (array([3. , 1.5]), True)
    """
    cube = np.asarray(cube)
    if dtype is None:
        dtype = np.float32 if cube.dtype == np.float32 else np.float64
    dtype = np.dtype(dtype).type
    if dtype not in (np.float64, np.float32):
        raise ValueError(f"dtype must be float64 or float32, not {dtype.__name__}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, not {target!r}")

    rows, cols, m = cube.shape
    n_pixels = rows * cols
    p0 = np.ascontiguousarray(p0, dtype)
    x = np.ascontiguousarray(x, dtype)
    w = np.ones(m, dtype) if weights is None else np.ascontiguousarray(weights, dtype)
    if p0.ndim != 1:
        raise ValueError(f"p0 must be one-dimensional, not of shape {p0.shape}")
    if w.shape != (m,) or len(x) != m:
        raise ValueError(f"x and weights need {m} entries, one per sample")
    n = p0.shape[0]

    default = math.sqrt(np.finfo(dtype).eps)
    tols = [dtype(default if t is None else t) for t in (tol_g, tol_p, tol_c)]
    step = dtype(default if dp_ratio is None else dp_ratio)
    scalars = (np.int32(max_iter), *tols, dtype(lambda_0_factor), step)
    inputs = (np.ascontiguousarray(cube, dtype).reshape(n_pixels, m), x, w, p0)
    outputs = [
        np.empty((n_pixels, n), dtype),  # p
        np.empty((n_pixels, n, n), dtype),  # cov
        np.empty(n_pixels, dtype),  # chi2
        np.empty(n_pixels, np.int32),  # n_iter
        np.empty(n_pixels, np.bool_),  # converged
    ]
    p, cov, chi2, n_iter, converged = _compile(model, n, dtype, target)(
        device, inputs, outputs, scalars
    )
    return FitResult(
        p.reshape(rows, cols, n),
        cov.reshape(rows, cols, n, n),
        chi2.reshape(rows, cols),
        n_iter.reshape(rows, cols),
        converged.reshape(rows, cols),
    )
