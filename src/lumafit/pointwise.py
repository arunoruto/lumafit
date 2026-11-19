"""
Pixel-wise fitting with a pointwise model, on the CPU or a Vulkan GPU.

The solver in this module evaluates the model one sample at a time and
allocates no arrays of run-time size. That makes the same code compile for
CPU threads (:cite:p:`lam2015numba`) and, through numba-vulkan, for any GPU
with a Vulkan driver.
"""

import math

import numpy as np
from numba import njit, prange

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
        """Fit one curve in place of `p`; the algorithm of the array solver."""
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


def make_pixelwise_fitter(model, n_params, dtype=np.float64, target="cpu"):
    """
    Compile a pixel-wise Levenberg-Marquardt fit for a pointwise model.

    Unlike `lumafit.levenberg_marquardt_pixelwise`, the model computes one
    sample at a time. In exchange the fit runs on a GPU, compiles in a
    second or two and is a little faster on the CPU as well.

    Parameters
    ----------
    model : callable
        ``model(p, x) -> float``, a function compiled with ``numba.njit``
        that returns the model value for the parameters `p` (1D array of
        length `n_params`) at one sample ``x`` of the independent
        variable.
    n_params : int
        Number of parameters.
    dtype : {numpy.float64, numpy.float32}, optional
        Precision of the computation. Most GPUs are several times faster
        in float32, where results agree with float64 to about 1e-4.
    target : {'cpu', 'vulkan'}, optional
        Run on all Numba threads, or on a Vulkan device through
        numba-vulkan (``pip install lumafit[vulkan]``).

    Returns
    -------
    callable
        ``fit(p0, target_y_3d, x, weights_1d=None, max_iter=100,
        tol_g=None, tol_p=None, tol_c=None, lambda_0_factor=1e-2,
        dp_ratio=None, device=None)``. It fits every
        ``target_y_3d[r, c, :]`` and returns ``(p, cov, chi2, n_iter,
        converged)`` with the shapes of
        `lumafit.levenberg_marquardt_pixelwise`, in `dtype`. `x` holds
        one entry per sample along its first axis. The tolerances and
        `dp_ratio` default to the square root of the machine precision
        of `dtype`. `device` selects the Vulkan device by index or name;
        the default is the first discrete GPU.

    Raises
    ------
    ValueError
        If `target` or `dtype` is not one of the supported values.

    See Also
    --------
    lumafit.levenberg_marquardt_pixelwise : The same fit for array models.

    Notes
    -----
    The algorithm is that of `lumafit.levenberg_marquardt_core` with
    Marquardt damping and a forward difference Jacobian; an analytical
    Jacobian is not supported. Pixels that contain a NaN are skipped.

    In a float32 fit every operation of the model has to stay in
    float32: a Python literal such as ``0.5 * p[0]`` promotes the
    expression to float64, which numba-vulkan reports with a warning.
    Use typed constants (``HALF = np.float32(0.5)``) instead.

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from numba import njit
    >>> from lumafit import make_pixelwise_fitter
    >>> @njit
    ... def decay(p, t):
    ...     return p[0] * math.exp(-t / p[1])
    >>> t = np.linspace(0.0, 5.0, 50)
    >>> cube = np.tile(3.0 * np.exp(-t / 1.5), (4, 4, 1))
    >>> fit = make_pixelwise_fitter(decay, 2)
    >>> p, cov, chi2, n_iter, converged = fit(np.array([1.0, 1.0]), cube, t)
    >>> np.round(p[0, 0], 6), bool(converged.all())
    (array([3. , 1.5]), True)
    """
    dtype = np.dtype(dtype).type
    if dtype not in (np.float64, np.float32):
        raise ValueError(f"dtype must be float64 or float32, not {dtype.__name__}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, not {target!r}")

    n = int(n_params)
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

    else:
        import numba_vulkan as nv

        @nv.jit
        def kernel(
            Y, x, w, p0, P, cov, chi2, n_iter, conv, max_iter, tg, tp, tc, lam, h
        ):
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
            on_device = [
                nv.device_array(o.shape, o.dtype, device=device) for o in outputs
            ]
            kernel.forall(inputs[0].shape[0], device=device)(
                *(nv.to_device(i, device=device) for i in inputs), *on_device, *scalars
            )
            return [d.copy_to_host(o) for d, o in zip(on_device, outputs, strict=True)]

    default_tol = math.sqrt(np.finfo(dtype).eps)

    def fit(
        p0,
        target_y_3d,
        x,
        weights_1d=None,
        max_iter=100,
        tol_g=None,
        tol_p=None,
        tol_c=None,
        lambda_0_factor=1e-2,
        dp_ratio=None,
        device=None,
    ):
        rows, cols, m = target_y_3d.shape
        n_pixels = rows * cols
        Y = np.ascontiguousarray(target_y_3d, dtype).reshape(n_pixels, m)
        w = (
            np.ones(m, dtype)
            if weights_1d is None
            else np.ascontiguousarray(weights_1d, dtype)
        )
        p0 = np.ascontiguousarray(p0, dtype)
        if p0.shape != (n,):
            raise ValueError(f"p0 must have shape ({n},), not {p0.shape}")
        if w.shape != (m,) or len(x) != m:
            raise ValueError(f"x and weights_1d need {m} entries, one per sample")

        outputs = [
            np.empty((n_pixels, n), dtype),  # p
            np.empty((n_pixels, n, n), dtype),  # cov
            np.empty(n_pixels, dtype),  # chi2
            np.empty(n_pixels, np.int32),  # n_iter
            np.empty(n_pixels, np.bool_),  # converged
        ]
        tols = [dtype(default_tol if t is None else t) for t in (tol_g, tol_p, tol_c)]
        step = dtype(default_tol if dp_ratio is None else dp_ratio)
        scalars = (np.int32(max_iter), *tols, dtype(lambda_0_factor), step)
        inputs = (Y, np.ascontiguousarray(x, dtype), w, p0)
        P, cov, chi2, n_iter, conv = launch(device, inputs, outputs, scalars)
        return (
            P.reshape(rows, cols, n),
            cov.reshape(rows, cols, n, n),
            chi2.reshape(rows, cols),
            n_iter.reshape(rows, cols),
            conv.reshape(rows, cols),
        )

    return fit
