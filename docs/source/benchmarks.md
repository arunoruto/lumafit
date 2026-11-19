# Benchmarks

How long does it take to fit every pixel of a 128 × 128 image cube?

```{image} benchmarks/benchmark-light.svg
:class: only-light
:alt: Time per fit for lumafit, SciPy and optimistix
```

```{image} benchmarks/benchmark-dark.svg
:class: only-dark
:alt: Time per fit for lumafit, SciPy and optimistix
```

## Results

```{include} benchmarks/results.md
```

## What is measured

Two models with four parameters each, sampled at 100 points per pixel:

- **Double exponential decay**, $p_0 e^{-t/p_1} + p_2 e^{-t/p_3}$
- **Polarization phase curve**,
  $p_0 \sin^{p_1}\!\alpha \, \cos^{p_2}(\alpha/2) \, \sin(\alpha - p_3)$

Every pixel has its own true parameters (±10 % around a common value) plus
Gaussian noise, and all solvers start from the same initial guess and use their
default tolerances.

- **SciPy** is {py:func}`scipy.optimize.least_squares` with `method="lm"`
  (MINPACK), called once per pixel from a Python loop on a plain NumPy model.
- **optimistix** is `optimistix.LevenbergMarquardt` under {py:func}`jax.vmap`
  on the CPU, in double precision. It could not be tested on a GPU here.
- **lumafit 0.2.3** is the previous release from PyPI, run with the same
  dependency versions.

Times are the best of three runs after a warm-up call. The warm-up is reported
separately as *first-call JIT*: lumafit compiles the solver for each new model
once per process, which takes around ten seconds and is the price for the speed
afterwards. If you only fit a few hundred curves, SciPy finishes first.

All solvers land on the same minimum; the last column is the median relative
difference of the fitted parameters to SciPy's.

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
