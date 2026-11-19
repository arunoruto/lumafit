# lumafit

A [Numba](https://numba.pydata.org/)-accelerated {cite:p}`lam2015numba`
Levenberg-Marquardt solver {cite:p}`levenberg1944method,marquardt1963algorithm`
for fitting many small curves at once, for example one curve per pixel of an
image cube.

- **Fast**: around 100× faster than looping over
  {py:func}`scipy.optimize.least_squares` {cite:p}`virtanen2020scipy` on 16
  threads, see [](benchmarks.md).
- **Small**: two functions, {py:func}`lumafit.levenberg_marquardt_core` for a
  single curve and {py:func}`lumafit.levenberg_marquardt_pixelwise` for a cube.
- **On GPUs too**: with a pointwise model the same fit runs on any GPU with a
  Vulkan driver through numba-vulkan {cite:p}`arnaut2026numbavulkan`, see
  [](usage.md).
- **Callable from compiled code**: the solver is a nopython function, so it can
  be used inside your own Numba-compiled loops.

```{toctree}
:maxdepth: 2
:caption: Contents

usage
benchmarks
references
```
