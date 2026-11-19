"""The result type shared by all solvers."""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class FitResult(NamedTuple):
    """
    Result of a fit.

    A named tuple, so it can also be unpacked as
    ``p, cov, chi2, n_iter, converged = fit_curve(...)``. For a cube
    every field has the two leading axes ``(rows, cols)``.

    Attributes
    ----------
    p : numpy.ndarray
        Fitted parameters, shape ``(n,)``. NaN for skipped pixels.
    cov : numpy.ndarray
        ``(J^T W J)^-1``, shape ``(n, n)``: the parameter covariance if
        the weights are inverse variances. Filled with ``inf`` if the
        matrix is singular or there are not more samples than
        parameters, and with NaN for skipped pixels.
    chi2 : float or numpy.ndarray
        Final weighted sum of squared residuals. NaN for skipped pixels.
    n_iter : int or numpy.ndarray
        Number of iterations performed. 0 for skipped pixels.
    converged : bool or numpy.ndarray
        Whether one of the tolerances was met within ``max_iter``. False
        for skipped pixels.
    """

    p: npt.NDArray[np.floating]
    cov: npt.NDArray[np.floating]
    chi2: float | npt.NDArray[np.floating]
    n_iter: int | npt.NDArray[np.integer]
    converged: bool | npt.NDArray[np.bool_]
