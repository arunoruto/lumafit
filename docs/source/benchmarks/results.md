16,384 fits per run (100 samples, 4 parameters each) on an Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 threads) with Intel UHD Graphics 630 and NVIDIA TITAN X; Python 3.12.7, NumPy 2.5.5, Numba 0.68.1, SciPy 1.18.1, numba-vulkan 0.1.2, JAX 0.12.1, optimistix 0.1.0.

### Double exponential decay

| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |
|---|--:|--:|--:|--:|--:|--:|
| SciPy least_squares, Python loop | 13.763 s | 840 µs | 1× | - | - | - |
| optimistix + jax.vmap, CPU | 3.358 s | 205 µs | 4× | - | 0.5 s | 4e-09 |
| lumafit 0.2.3, 1 thread | 1.575 s | 96.1 µs | 9× | 9.8 | 15.2 s | - |
| **lumafit (this version), array model, 1 thread** | 0.823 s | 50.2 µs | 17× | 5.6 | 13.8 s | 2e-08 |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X | 0.432 s | 26.4 µs | 32× | - | 0.6 s | 4e-09 |
| lumafit 0.2.3, 16 threads | 0.252 s | 15.4 µs | 55× | 9.8 | - | - |
| lumafit 0.2.3, 16 threads, analytic J | 0.193 s | 11.8 µs | 71× | 9.8 | 10.5 s | - |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X, float32 | 0.155 s | 9.5 µs | 89× | - | 0.7 s | 5e-06 |
| **lumafit (this version), array model, 16 threads** | 0.137 s | 8.3 µs | 101× | 5.6 | - | 2e-08 |
| **lumafit (this version), pointwise, 16 threads** | 0.114 s | 7.0 µs | 121× | 5.5 | 2.0 s | 2e-08 |
| **lumafit (this version), array model, 16 threads, analytic J** | 0.113 s | 6.9 µs | 122× | 5.6 | 9.4 s | 2e-08 |
| **lumafit (this version), pointwise, Vulkan on Intel UHD Graphics 630, float32** | 0.056 s | 3.4 µs | 246× | 3.6 | 1.4 s | 5e-05 |
| **lumafit (this version), pointwise, 16 threads, float32** | 0.051 s | 3.1 µs | 269× | 3.6 | 1.9 s | 5e-05 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X** | 0.046 s | 2.8 µs | 302× | 5.5 | 1.6 s | 2e-08 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X, float32** | 0.008 s | 0.51 µs | 1643× | 3.6 | 1.4 s | 5e-05 |

### Polarization phase curve

| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |
|---|--:|--:|--:|--:|--:|--:|
| SciPy least_squares, Python loop | 16.114 s | 984 µs | 1× | - | - | - |
| optimistix + jax.vmap, CPU | 7.391 s | 451 µs | 2× | - | 0.5 s | 2e-08 |
| lumafit 0.2.3, 1 thread | 4.039 s | 246 µs | 4× | 8.8 | 10.4 s | - |
| **lumafit (this version), array model, 1 thread** | 2.600 s | 159 µs | 6× | 6.1 | 9.0 s | 5e-08 |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X | 0.925 s | 56.5 µs | 17× | - | 0.6 s | 2e-08 |
| lumafit 0.2.3, 16 threads | 0.529 s | 32.3 µs | 30× | 8.8 | - | - |
| **lumafit (this version), pointwise, 16 threads** | 0.365 s | 22.3 µs | 44× | 6.1 | 1.8 s | 6e-08 |
| **lumafit (this version), array model, 16 threads** | 0.358 s | 21.9 µs | 45× | 6.1 | - | 5e-08 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X** | 0.323 s | 19.7 µs | 50× | 6.1 | 2.5 s | 6e-08 |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X, float32 | 0.273 s | 16.6 µs | 59× | - | 0.6 s | 3e-06 |
| **lumafit (this version), pointwise, Vulkan on Intel UHD Graphics 630, float32** | 0.238 s | 14.5 µs | 68× | 4.0 | 2.1 s | 2e-04 |
| **lumafit (this version), pointwise, 16 threads, float32** | 0.155 s | 9.5 µs | 104× | 4.0 | 2.0 s | 2e-04 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X, float32** | 0.017 s | 1.1 µs | 928× | 4.0 | 2.2 s | 2e-04 |
