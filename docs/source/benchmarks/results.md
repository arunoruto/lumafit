16,384 fits per run (100 samples, 4 parameters each) on an Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 threads); Python 3.12.7, NumPy 2.5.5, Numba 0.68.1, SciPy 1.18.1.

### Double exponential decay

| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |
|---|--:|--:|--:|--:|--:|--:|
| scipy least_squares (lm), loop | 13.970 s | 852.7 µs | 1.0× | - | - | - |
| optimistix LM + jax.vmap (CPU) | 3.365 s | 205.4 µs | 4.2× | - | 0.5 s | 4e-09 |
| lumafit 0.2.3, 1 thread | 1.575 s | 96.1 µs | 8.9× | 9.8 | 15.2 s | - |
| **lumafit (this version), 1 thread** | 0.826 s | 50.4 µs | 16.9× | 5.6 | 13.7 s | 2e-08 |
| lumafit 0.2.3, 16 threads | 0.252 s | 15.4 µs | 55.4× | 9.8 | - | - |
| lumafit 0.2.3, 16 threads, analytic J | 0.193 s | 11.8 µs | 72.3× | 9.8 | 10.5 s | - |
| **lumafit (this version), 16 threads** | 0.138 s | 8.4 µs | 100.9× | 5.6 | - | 2e-08 |
| **lumafit (this version), 16 threads, analytic J** | 0.112 s | 6.8 µs | 125.1× | 5.6 | 9.3 s | 2e-08 |

### Polarization phase curve

| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |
|---|--:|--:|--:|--:|--:|--:|
| scipy least_squares (lm), loop | 16.262 s | 992.5 µs | 1.0× | - | - | - |
| optimistix LM + jax.vmap (CPU) | 7.392 s | 451.2 µs | 2.2× | - | 0.6 s | 2e-08 |
| lumafit 0.2.3, 1 thread | 4.039 s | 246.5 µs | 4.0× | 8.8 | 10.4 s | - |
| **lumafit (this version), 1 thread** | 2.589 s | 158.0 µs | 6.3× | 6.1 | 8.9 s | 5e-08 |
| lumafit 0.2.3, 16 threads | 0.529 s | 32.3 µs | 30.8× | 8.8 | - | - |
| **lumafit (this version), 16 threads** | 0.353 s | 21.5 µs | 46.1× | 6.1 | - | 5e-08 |
