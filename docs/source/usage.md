# Usage

## Installation

```bash
pip install lumafit
# or
uv add lumafit
```

## The model function

The model is called as `func(p, *args)` and returns the model values as a 1D
`float64` array. It has to be a Numba nopython function **defined with
`*args`**; whatever you pass as `args` to the solver arrives there, typically
the independent variable first.

```python
import numpy as np
from numba import njit


@njit
def decay(p, *args):
    t = args[0]
    return p[0] * np.exp(-t / p[1]) + p[2] * np.exp(-t / p[3])
```

## Fitting a single curve

```python
from lumafit import levenberg_marquardt_core

t = np.linspace(0.1, 25, 100)
p_true = np.array([5.0, 2.0, 2.0, 10.0])
y = decay(p_true, t) + np.random.default_rng(0).normal(0, 0.02, t.shape)

p0 = np.array([4.0, 1.5, 1.5, 8.0])  # initial guess

p_fit, cov, chi2, n_iter, converged = levenberg_marquardt_core(
    decay, p0, y, args=(t,)
)
```

`cov` is $(J^T W J)^{-1}$. If `weights` are inverse variances it is the
parameter covariance; for an unweighted fit scale it by the reduced chi-squared
{cite:p}`gavin2020levenberg`:

```python
sigma = np.sqrt(np.diag(cov) * chi2 / (y.size - p_fit.size))
```

## Fitting an image cube

{py:func}`lumafit.levenberg_marquardt_pixelwise` fits every
`cube[row, col, :]` independently and in parallel. Pixels containing a NaN are
skipped, so NaN doubles as a mask.

```python
from lumafit import levenberg_marquardt_pixelwise

cube = np.tile(y, (128, 128, 1))  # (rows, cols, samples)

p, cov, chi2, n_iter, converged = levenberg_marquardt_pixelwise(
    decay, p0, cube, args_for_each_pixel=(t,)
)
p.shape  # (128, 128, 4)
```

The number of threads is set with {py:func}`numba.set_num_threads` or the
`NUMBA_NUM_THREADS` environment variable.

```{note}
The first call compiles the solver for your model, which takes around ten
seconds. Later calls in the same process run at full speed.
```

## Weights

`weights` (or `weights_1d` for a cube) are multiplied with the squared
residuals, so pass `1 / sigma**2`:

```python
sigma = np.full(t.shape, 0.02)
levenberg_marquardt_core(decay, p0, y, weights=1 / sigma**2, args=(t,))
```

## Analytical Jacobian

By default the Jacobian is approximated with forward differences
{cite:p}`nocedal2006numerical`, costing one model evaluation per parameter and
iteration. If you can write it down, pass it
as `jac_func`; it takes the same arguments as the model and returns an array of
shape `(samples, parameters)`:

```python
@njit
def decay_jac(p, *args):
    t = args[0]
    J = np.empty((t.shape[0], 4))
    e1 = np.exp(-t / p[1])
    e2 = np.exp(-t / p[3])
    J[:, 0] = e1
    J[:, 1] = p[0] * e1 * t / p[1] ** 2
    J[:, 2] = e2
    J[:, 3] = p[2] * e2 * t / p[3] ** 2
    return J


levenberg_marquardt_core(decay, p0, y, jac_func=decay_jac, args=(t,))
```

## Minimizing residuals directly

Without `target_y`, the output of the function is taken as the residual vector,
as in {py:func}`scipy.optimize.least_squares` {cite:p}`virtanen2020scipy`:

```python
@njit
def residuals(p, *args):
    t, y = args
    return p[0] * np.exp(-t / p[1]) - y


levenberg_marquardt_core(residuals, np.array([1.0, 1.0]), args=(t, y))
```

## Pointwise models and GPUs

{py:func}`lumafit.make_pixelwise_fitter` is a second solver for image cubes.
Its model computes **one sample at a time** instead of a whole curve:

```python
import math
from lumafit import make_pixelwise_fitter


@njit
def decay_point(p, t):
    return p[0] * math.exp(-t / p[1]) + p[2] * math.exp(-t / p[3])


fit = make_pixelwise_fitter(decay_point, n_params=4)
p, cov, chi2, n_iter, converged = fit(p0, cube, t)
```

The algorithm and the results are the same, but nothing in it allocates memory
while it runs. It therefore compiles in about two seconds instead of ten, is a
little faster on the CPU, and can run on a GPU.

### On a GPU

With [numba-vulkan](https://github.com/arunoruto/numba-vulkan)
{cite:p}`arnaut2026numbavulkan` the same fit runs on any GPU that has a Vulkan
driver, whether it is from NVIDIA, AMD, Intel or Apple:

```bash
pip install "lumafit[vulkan]"
```

```python
fit = make_pixelwise_fitter(decay_point, 4, dtype=np.float32, target="vulkan")
p, cov, chi2, n_iter, converged = fit(p0, cube, t)             # first discrete GPU
p, cov, chi2, n_iter, converged = fit(p0, cube, t, device="intel")  # by name or index
```

Arrays go in and come out as NumPy arrays; the copies to and from the device
are part of the call. See [](benchmarks.md) for what to expect.

### float32

Most GPUs compute several times faster in `float32` than in `float64`, and some
have no `float64` at all. `dtype=np.float32` runs the whole fit in single
precision, on the CPU as well. Two things change:

- The fitted parameters agree with a `float64` fit to about 1e-4, because the
  finite difference Jacobian and the convergence tests are limited by the
  precision. The tolerances default to the square root of the machine
  precision of `dtype`.
- Every operation in the model has to stay in `float32`. A Python literal
  promotes the expression to `float64`, which numba-vulkan reports with a
  warning. Use typed constants:

  ```python
  HALF = np.float32(0.5)


  @njit
  def model(p, t):
      return p[0] * math.cos(HALF * t)  # not: 0.5 * t
  ```

### Differences to the array solver

- The model signature is `model(p, x)`, where `x` is one entry of the
  independent variable (`t[k]`, or a row of it if `t` is 2D).
- There is no `jac_func`; the Jacobian always comes from forward differences.
- Damping is always Marquardt's.
- `n_iter` is `int32` and all floating point results have the chosen `dtype`.

## Algorithm

The solver is a damped Gauss-Newton iteration
{cite:p}`levenberg1944method,marquardt1963algorithm`. Each step solves

$$
\left(J^T W J + \lambda \, \operatorname{diag}(J^T W J)\right) \delta = J^T W r
$$

for the parameter update $\delta$, with residuals $r$ and weights $W$. The
damping $\lambda$ is adapted from the ratio of the actual to the predicted
decrease of $\chi^2$ {cite:p}`nielsen1999damping`. A fit has converged when the
gradient, the relative parameter change or the relative decrease of $\chi^2$
falls below `tol_g`, `tol_p` or `tol_c`. See
{cite:t}`nocedal2006numerical` and {cite:t}`gavin2020levenberg` for background,
and {cite:t}`more1978levenbergmarquardt` for the trust-region formulation used
by MINPACK.
