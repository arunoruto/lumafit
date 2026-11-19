16,384 fits per run (100 samples, 4 parameters each) on an Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 threads) with Intel UHD Graphics 630 and NVIDIA TITAN X; Python 3.12.7, NumPy 2.5.5, Numba 0.68.1, SciPy 1.18.1, numba-vulkan 0.1.2, JAX 0.12.1, optimistix 0.1.0.

### Double exponential decay

| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |
|---|--:|--:|--:|--:|--:|--:|
| SciPy least_squares, Python loop | 14.602 s | 891 µs | 1× | - | - | - |
| optimistix + jax.vmap, CPU | 3.371 s | 206 µs | 4× | - | 0.5 s | 4e-09 |
| lumafit 0.2.3, 1 thread | 1.575 s | 96.1 µs | 9× | 9.8 | 15.2 s | - |
| **lumafit (this version), array model, 1 thread** | 0.820 s | 50.1 µs | 18× | 5.6 | 13.9 s | 2e-08 |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X | 0.433 s | 26.5 µs | 34× | - | 0.7 s | 4e-09 |
| lumafit 0.2.3, 16 threads | 0.252 s | 15.4 µs | 58× | 9.8 | - | - |
| lumafit 0.2.3, 16 threads, analytic J | 0.193 s | 11.8 µs | 76× | 9.8 | 10.5 s | - |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X, float32 | 0.156 s | 9.5 µs | 94× | - | 0.5 s | 5e-06 |
| **lumafit (this version), array model, 16 threads** | 0.137 s | 8.4 µs | 107× | 5.6 | - | 2e-08 |
| **lumafit (this version), pointwise, 16 threads** | 0.118 s | 7.2 µs | 124× | 5.5 | 2.1 s | 2e-08 |
| **lumafit (this version), array model, 16 threads, analytic J** | 0.115 s | 7.0 µs | 127× | 5.6 | 9.0 s | 2e-08 |
| **lumafit (this version), pointwise, Vulkan on Intel UHD Graphics 630, float32** | 0.055 s | 3.3 µs | 268× | 3.6 | 0.7 s | 5e-05 |
| **lumafit (this version), pointwise, 16 threads, float32** | 0.052 s | 3.2 µs | 280× | 3.6 | 2.0 s | 5e-05 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X** | 0.046 s | 2.8 µs | 320× | 5.5 | 0.8 s | 2e-08 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X, float32** | 0.009 s | 0.53 µs | 1684× | 3.6 | 0.6 s | 5e-05 |

### Polarization phase curve

| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |
|---|--:|--:|--:|--:|--:|--:|
| SciPy least_squares, Python loop | 16.999 s | 1038 µs | 1× | - | - | - |
| optimistix + jax.vmap, CPU | 7.388 s | 451 µs | 2× | - | 0.6 s | 2e-08 |
| lumafit 0.2.3, 1 thread | 4.039 s | 246 µs | 4× | 8.8 | 10.4 s | - |
| **lumafit (this version), array model, 1 thread** | 2.612 s | 159 µs | 7× | 6.1 | 8.9 s | 5e-08 |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X | 0.925 s | 56.5 µs | 18× | - | 0.7 s | 2e-08 |
| lumafit 0.2.3, 16 threads | 0.529 s | 32.3 µs | 32× | 8.8 | - | - |
| **lumafit (this version), pointwise, 16 threads** | 0.374 s | 22.8 µs | 45× | 6.1 | 2.1 s | 6e-08 |
| **lumafit (this version), array model, 16 threads** | 0.344 s | 21.0 µs | 49× | 6.1 | - | 5e-08 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X** | 0.326 s | 19.9 µs | 52× | 6.1 | 0.8 s | 6e-08 |
| optimistix + jax.vmap, CUDA on NVIDIA TITAN X, float32 | 0.273 s | 16.6 µs | 62× | - | 0.6 s | 3e-06 |
| **lumafit (this version), pointwise, Vulkan on Intel UHD Graphics 630, float32** | 0.237 s | 14.5 µs | 72× | 4.0 | 0.7 s | 2e-04 |
| **lumafit (this version), pointwise, 16 threads, float32** | 0.151 s | 9.2 µs | 112× | 4.0 | 1.8 s | 2e-04 |
| **lumafit (this version), pointwise, Vulkan on NVIDIA TITAN X, float32** | 0.018 s | 1.1 µs | 966× | 4.0 | 0.8 s | 2e-04 |
