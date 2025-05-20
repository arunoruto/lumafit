# Usage Guide

This section explains how to install and use Lumafit for your fitting tasks.

## Installation

Install Lumafit using pip:

```bash
pip install lumafit
```

## Basic Curve Fitting Example

Here is a simple example demonstrating how to use :py:func:`lumafit.levenberg_marquardt_core`:

```python
import numpy as np
from lumafit import levenberg_marquardt_core
# Assuming model_exponential_decay is accessible, e.g., imported in __init__.py
from lumafit import model_exponential_decay

# Define some dummy data based on the exponential decay model
t = np.linspace(0, 10, 50, dtype=np.float64)
p_true = np.array([10.0, 2.0, 5.0, 0.5], dtype=np.float64)
# Add some noise
rng = np.random.default_rng(42)
y_data = model_exponential_decay(t, p_true) + rng.normal(0, 0.5, size=t.shape).astype(np.float64)

# Initial guess for parameters
p0 = np.array([8.0, 1.5, 6.0, 1.0], dtype=np.float64)

# Perform the fit
p_fit, cov_p, chi2, iters, converged = levenberg_marquardt_core(
    model_exponential_decay,
    t,
    y_data,
    p0,
    max_iter=500 # Example of passing optional arguments
)

print(f"Initial guess: {p0}")
print(f"True parameters: {p_true}")
print(f"Fitted parameters: {p_fit}")
print(f"Fit converged: {converged}")
print(f"Chi-squared: {chi2:.2f}")
print(f"Iterations: {iters}")
```

## Pixel-wise Fitting

For 3D image data, use :py:func:`lumafit.levenberg_marquardt_pixelwise`:

```python
import numpy as np
from lumafit import levenberg_marquardt_pixelwise
from lumafit import model_exponential_decay

# Create dummy 3D data (e.g., 2x2 pixels, 50 data points per pixel)
rows, cols, depth = 2, 2, 50
t = np.linspace(0, 10, depth, dtype=np.float64)
p_true_base = np.array([10.0, 2.0, 5.0, 0.5], dtype=np.float64)
p0_global = np.array([8.0, 1.5, 6.0, 1.0], dtype=np.float64)
y_data_3d = np.empty((rows, cols, depth), dtype=np.float64)

rng = np.random.default_rng(43) # New seed for pixel data

for r in range(rows):
    for c in range(cols):
        # Slightly vary true params per pixel
        p_pixel_true = p_true_base * (1 + rng.uniform(-0.05, 0.05, size=p_true_base.shape))
        y_clean_pixel = model_exponential_decay(t, p_pixel_true)
        noise_pixel = rng.normal(0, 0.2, size=depth).astype(np.float64) # Add noise
        y_data_3d[r, c, :] = y_clean_pixel + noise_pixel

# Perform pixel-wise fit
p_results, cov_results, chi2_results, n_iter_results, converged_results = levenberg_marquardt_pixelwise(
    model_exponential_decay,
    t,
    y_data_3d,
    p0_global,
    max_iter=200
)

print("Fitted parameters (first pixel):", p_results[0, 0, :])
print("Convergence status (all pixels):\n", converged_results)
print("Chi-squared values (all pixels):\n", chi2_results)
```
