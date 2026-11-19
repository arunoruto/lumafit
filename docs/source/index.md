# lumafit

A [Numba](https://numba.pydata.org/)-accelerated Levenberg-Marquardt solver for
fitting many small curves at once, for example one curve per pixel of an image
cube.

- **Fast**: around 100× faster than looping over
  {py:func}`scipy.optimize.least_squares` on 16 threads, see
  [](benchmarks.md).
- **Small**: two functions, {py:func}`lumafit.levenberg_marquardt_core` for a
  single curve and {py:func}`lumafit.levenberg_marquardt_pixelwise` for a cube.
- **Callable from compiled code**: the solver is a nopython function, so it can
  be used inside your own Numba-compiled loops.

```{toctree}
:maxdepth: 2
:caption: Contents

usage
benchmarks
references
```
