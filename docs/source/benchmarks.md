# Benchmarks

How long does it take to fit every pixel of a 128 × 128 image cube?

```{image} benchmarks/benchmark-light.svg
:class: light-only
:alt: Time per fit for lumafit, SciPy and optimistix on CPU and GPU
```

```{image} benchmarks/benchmark-dark.svg
:class: dark-only
:alt: Time per fit for lumafit, SciPy and optimistix
```

## Results

```{include} benchmarks/results.md
```

## What is measured

Two models with four parameters each, sampled at 100 points per pixel:

- **Double exponential decay**, $p_0 e^{-t/p_1} + p_2 e^{-t/p_3}$
- **Polarization phase curve**,
  $p_0 \sin^{p_1}\!\alpha \, \cos^{p_2}(\alpha/2) \, \sin(\alpha - p_3)$,
  the trigonometric function of {cite:t}`lumme1993twoparameter` that is widely
  used for the polarization of asteroids and comets
  {cite:p}`penttila2005statistical`

Every pixel has its own true parameters (±10 % around a common value) plus
Gaussian noise, and all solvers start from the same initial guess and use their
default tolerances. Every time is end to end: NumPy arrays of the solver's
precision go in and NumPy arrays come out, so copies to and from a GPU are
included.

- **SciPy** {cite:p}`virtanen2020scipy` is
  {py:func}`scipy.optimize.least_squares` with `method="lm"`, which wraps the
  MINPACK implementation {cite:p}`more1978levenbergmarquardt`. It is called
  once per pixel from a Python loop on a plain NumPy {cite:p}`harris2020array`
  model.
- **optimistix** {cite:p}`rader2024optimistix` is
  `optimistix.LevenbergMarquardt` under `jax.vmap` {cite:p}`bradbury2018jax` on
  the CPU, in double precision. It could not be tested on a GPU here.
- **optimistix on CUDA** is the same code on the GPU, in double and in single
  precision.
- **lumafit, array model** is {py:func}`lumafit.levenberg_marquardt_pixelwise`.
- **lumafit, pointwise** is {py:func}`lumafit.make_pixelwise_fitter` on the CPU
  threads or, as **Vulkan on ...**, on a GPU through numba-vulkan
  {cite:p}`arnaut2026numbavulkan`. The integrated Intel GPU was only run in
  single precision.
- **lumafit 0.2.3** is the previous release from PyPI, run with the same
  dependency versions.

Times are the best of three runs after a warm-up call. The warm-up is reported
separately as *first-call JIT*: lumafit compiles the solver for each new model
once per process with Numba {cite:p}`lam2015numba`. That takes around ten
seconds for an array model and one to two seconds for a pointwise model, and is
the price for the speed afterwards. If you only fit a few hundred curves, SciPy
finishes first.

All solvers land on the same minimum; the last column is the median relative
difference of the fitted parameters to SciPy's. In double precision it is 1e-7
or less. In single precision lumafit is at about 1e-4 and optimistix at about
1e-5: lumafit's finite difference Jacobian is limited by the precision, while
JAX differentiates the model exactly.

## On the GPU

On the same NVIDIA TITAN X, lumafit through Vulkan is about 15 to 18 times
faster than optimistix through CUDA in single precision, and 3 to 9 times
faster in double precision. A likely reason is how the fits are batched rather
than the drivers: under `jax.vmap` every pixel keeps iterating until the
slowest one has converged, and each iteration is a sequence of batched array
operations, whereas a lumafit kernel runs one complete fit per GPU thread and
stops when that fit is done.

Double precision is expensive on this GPU, most of all for the polarization
curve: Vulkan's built-in math functions are single precision only, so the
double precision power functions come from a software library (libclc). There
the GPU in `float64` is no faster than 16 CPU threads, and the choice is
between double precision and a run that is about 20 times shorter.

The integrated Intel GPU comes close to the 16 CPU threads in single precision:
it is as fast for the decay and 1.6 times slower for the polarization curve.
That is the point of the Vulkan target; it runs wherever there is a Vulkan
driver.

A 512 × 512 cube (262,144 fits, `--side 512`) gives the same ordering: 0.46 µs
and 0.66 µs per fit for lumafit on the TITAN X in single precision, against
10 µs and 23 µs for optimistix.

## What changed since 0.2.3

- The damping parameter is updated with Nielsen's gain-ratio rule
  {cite:p}`nielsen1999damping`, and its starting value is no longer scaled by
  the size of $J^T W J$ when Marquardt damping already does that. Fits now need
  around 6 iterations instead of 9 to 10, about as many as MINPACK.
- The Jacobian is not re-evaluated after the last step.

A hand-written Cholesky solve and explicit loops for $J^T W J$ were also tried
in place of BLAS/LAPACK. They gained 3 to 17 % depending on model and thread
count and were left out for now: the model evaluations dominate the run time.

## Reproducing

```bash
uv run --group bench python benchmarks/run.py     # writes benchmarks/results.json
uv run --group bench python benchmarks/report.py  # writes the table and figures
```

Solvers whose package or device is missing are skipped. The Vulkan rows need a
Vulkan loader and driver, and the CUDA rows a CUDA build of JAX:

```bash
uv run --group bench --with "jax[cuda12]" python benchmarks/run.py
```
