<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!-- Add shields relevant to your project, e.g., build status, PyPI version -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

<!-- Optional: [![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- Replace with your project logo if you have one -->
  <!-- <a href="https://github.com/arunoruto/lumafit]"><img src="images/logo.png" alt="Logo" width="80" height="80"></a> -->

  <h3 align="center">lumafit</h3>

  <p align="center">
    A Numba-accelerated Levenberg-Marquardt fitting library for Python.
    <br />
    Optimized for pixel-wise fitting on 3D image data.
    <br />
    <br />
    <!-- Optional: Add links to documentation if you create it later -->
    <!-- <a href="https://github.com/arunoruto/lumafit]"><strong>Explore the docs »</strong></a> -->
    <br />
    <a href="https://github.com/arunoruto/lumafit/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/arunoruto/lumafit/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#tests">Tests</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This library provides a high-performance implementation of the Levenberg-Marquardt (LM) algorithm for non-linear least squares fitting, accelerated using [Numba](https://numba.pydata.org/).

The primary motivation for `lumafit` is to efficiently perform fitting tasks on large multi-dimensional datasets, such as fitting a curve along the third dimension for every pixel in a 3D image stack. Numba's Just-In-Time (JIT) compilation and parallel processing capabilities (`numba.prange`) are leveraged to drastically reduce computation time compared to pure Python implementations.

Key features:

- Core Levenberg-Marquardt algorithm (`levenberg_marquardt_core`).
- Specialized function for fitting curves pixel-wise on 3D data (`levenberg_marquardt_pixelwise`) with parallel execution.
- Support for weighted least squares.
- Numerical Jacobian calculation via finite differences (Numba-accelerated).
- Implementation based on standard LM algorithm formulations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][Python.any]][Python-url]
- [![Numba][Numba-shield]][Numba-url]
- [![NumPy][NumPy-shield]][NumPy-url]
- [![SciPy][SciPy-shield]][SciPy-url] (Used in tests for comparison)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To get a local copy of `lumafit` up and running, follow these simple steps.

### Prerequisites

You need Python 3.9+ installed. Using a virtual environment is recommended.
You will also need standard Python build tools, typically included with `pip`.

```sh
python -m venv venv
source venv/bin/activate # On Linux/macOS
# venv\Scripts\activate # On Windows
```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/arunoruto/lumafit.git
   cd lumafit
   ```
2. Install the package using `pip`, which will use the `pyproject.toml` file:
   ```sh
   pip install .
   ```
   If you want to install dependencies required for running tests, use the `[test]` extra:
   ```sh
   pip install .[test]
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

The library provides two main functions: `levenberg_marquardt_core` for fitting a single curve and `levenberg_marquardt_pixelwise` for fitting a 3D data stack.

First, you need to define your model function. **Your model function must be compatible with Numba's `nopython=True` mode.** This means it should primarily use NumPy functions and basic Python constructs supported by Numba.

```python
# Example model function (exponential decay + another exponential decay)
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def my_model(t, p):
    """
    My example non-linear model.

    Args:
        t (np.ndarray): Independent variable (1D array).
        p (np.ndarray): Parameters (1D array), e.g., [A1, tau1, A2, tau2].

    Returns:
        np.ndarray: Model output evaluated at t.
    """
    # Add checks for potential zero divisors if parameters are in denominators
    term1 = np.zeros_like(t, dtype=np.float64)
    if np.abs(p[1]) > 1e-12: # Avoid division by zero or near-zero
        term1 = p[0] * np.exp(-t / p[1])

    term2 = np.zeros_like(t, dtype=np.float64)
    if np.abs(p[3]) > 1e-12:
         term2 = p[2] * np.exp(-t / p[3])

    return term1 + term2

# Or the polarization model:
# @jit(nopython=True, cache=True)
# def polarization_model(t, p):
#     # ... (your polarization model code from tests/lmba.py)
#     t_rad = t * np.pi / 180.0
#     p3_rad = p[3] * np.pi / 180.0
#     sin_t_rad_arr = np.sin(t_rad)
#     term1_base = sin_t_rad_arr.copy()
#     mask_problematic_base = (np.abs(term1_base) < 1e-15) & (p[1] < 0.0) # Using a small epsilon
#     term1_base[mask_problematic_base] = 1e-15 # Replace near-zero with tiny number
#     term1 = np.power(term1_base, p[1])
#     term2 = np.power(np.cos(t_rad / 2.0), p[2])
#     term3 = np.sin(t_rad - p3_rad)
#     return p[0] * term1 * term2 * term3
```

### Fitting a single curve

If you have a single 1D array of data (`y_data`) corresponding to independent variable values (`t_data`), you can use `levenberg_marquardt_core`. This function is the core engine for minimizing the difference between your model and the data.

```python
import numpy as np
from lmba import levenberg_marquardt_core

# Assume my_model is defined and JIT-compiled above

# Generate some synthetic data (replace with your actual data)
t_data = np.linspace(0.1, 25, 100, dtype=np.float64)
p_true = np.array([5.0, 2.0, 2.0, 10.0], dtype=np.float64)
y_clean = my_model(t_data, p_true)
noise = np.random.default_rng(42).normal(0, 0.1, size=t_data.shape).astype(np.float64)
y_data = (y_clean + noise).astype(np.float64)

# Initial guess for parameters
p_initial = np.array([4.0, 1.5, 1.5, 8.0], dtype=np.float64)

# Optional: weights (e.g., inverse variance if noise std is known)
weights = 1.0 / (0.1**2 + np.finfo(float).eps) # Assuming noise_std = 0.1

# Run the fit
p_fit, cov, chi2, iters, conv = levenberg_marquardt_core(
    my_model,       # Your Numba-compiled model function
    t_data,         # Independent variable data (1D array)
    y_data,         # Dependent variable data (1D array)
    p_initial,      # Initial guess (1D array)
    weights=weights, # Optional weights (1D array or None)
    max_iter=1000,  # Max iterations
    tol_g=1e-7,     # Gradient tolerance
    tol_p=1e-7,     # Parameter change tolerance
    tol_c=1e-7,     # Chi-squared change tolerance
    # ... other optional parameters
)

print(f"Fit converged: {conv}")
print(f"Iterations: {iters}")
print(f"Final Chi-squared: {chi2}")
print(f"Fitted parameters: {p_fit}")
# print(f"Covariance matrix: {cov}") # Covariance can be large
```

### Fitting pixel-wise on 3D data

For a 3D NumPy array where each `(row, col)` location has a curve along the third dimension (`data_cube[row, col, :]`), use `levenberg_marquardt_pixelwise`. This function parallelizes the fitting process across the `row` and `col` dimensions using `numba.prange`.

```python
import numpy as np
from lmba import levenberg_marquardt_pixelwise

# Assume my_model is defined and JIT-compiled above

# Generate some synthetic 3D data (replace with your actual data)
rows, cols, depth = 100, 100, 50 # Example dimensions
t_data = np.linspace(0.1, 25, depth, dtype=np.float64)
data_cube = np.empty((rows, cols, depth), dtype=np.float64)

p_true_base = np.array([5.0, 2.0, 2.0, 10.0], dtype=np.float64)
rng = np.random.default_rng(42)

for r_idx in range(rows):
    for c_idx in range(cols):
        # Vary true params slightly per pixel
        p_pixel_true = p_true_base * (1 + rng.uniform(-0.05, 0.05, size=p_true_base.shape))
        y_clean_pixel = my_model(t_data, p_pixel_true)
        noise_pixel = rng.normal(0, 0.1, size=depth).astype(np.float64)
        data_cube[r_idx,c_idx,:] = (y_clean_pixel + noise_pixel).astype(np.float64)

# Global initial guess for all pixels
p0_global = np.array([4.0, 1.5, 1.5, 8.0], dtype=np.float64)

# Optional weights (applied to each pixel identically)
# weights_1d = 1.0 / (0.1**2 + np.finfo(float).eps) # Assuming noise_std = 0.1

# Run the pixel-wise fit (this is parallelized)
p_results, cov_results, chi2_results, n_iter_results, conv_results = levenberg_marquardt_pixelwise(
    my_model,        # Your Numba-compiled model function
    t_data,          # Independent variable (1D array, common for all pixels)
    data_cube,       # 3D data array (rows x cols x depth)
    p0_global,       # Global initial guess (1D array)
    # Optional parameters for the core LM algorithm, passed to each pixel fit
    # weights_1d=weights_1d,
    max_iter=500,
    tol_g=1e-6,
    tol_p=1e-6,
    tol_c=1e-6,
    # ... other optional parameters
)

print(f"Pixel-wise fitting finished.")
print(f"Shape of fitted parameters: {p_results.shape}") # (rows x cols x n_params)
print(f"Shape of convergence flags: {conv_results.shape}") # (rows x cols)
print(f"Percentage converged: {np.sum(conv_results) / (rows*cols) * 100.0:.2f}%")
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tests

The library includes a test suite using `pytest` to verify the correctness of the core LM algorithm and the pixel-wise function against known solutions (for noiseless data) and against `scipy.optimize.least_squares` (for noisy data).

To run the tests:

1.  Ensure you have installed the test dependencies: `pip install .[test]`
2.  Navigate to the project root directory in your terminal.
3.  Run pytest:
    ```sh
    pytest
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [ ] Add support for analytical Jacobian functions (instead of only finite differences).
- [ ] Implement parameter bounds.
- [ ] Investigate alternative damping strategies (e.g., Nielsen's method).
- [ ] Improve robustness for ill-conditioned problems.
- [ ] Add more detailed documentation and examples.
- [ ] Potentially publish on PyPI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are welcome! If you have suggestions or find bugs, please open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the [MIT License][license-url]. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Mirza Arnaut - mirza.arnaut@tu-dortmund.de

Project Link: https://github.com/arunoruto/lumafit

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- The original Levenberg-Marquardt algorithm (see references in `lmba/__init__.py`).
- [Numba](https://numba.pydata.org/) for providing the acceleration capabilities.
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for fundamental numerical computing tools.
- [pytest](https://docs.pytest.org/) for the testing framework.
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template) for the README structure template.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/arunoruto/lumafit.svg?style=for-the-badge
[contributors-url]: https://github.com/arunoruto/lumafit/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/arunoruto/lumafit.svg?style=for-the-badge
[forks-url]: https://github.com/arunoruto/lumafit/network/members
[stars-shield]: https://img.shields.io/github/stars/arunoruto/lumafit.svg?style=for-the-badge
[stars-url]: https://github.com/arunoruto/lumafit/stargazers
[issues-shield]: https://img.shields.io/github/issues/arunoruto/lumafit.svg?style=for-the-badge
[issues-url]: https://github.com/arunoruto/lumafit/issues
[license-shield]: https://img.shields.io/github/license/arunoruto/lumafit.svg?style=for-the-badge
[license-url]: https://github.com/arunoruto/lumafit/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/[your_linkedin_username]
[Python.any]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Numba-shield]: https://img.shields.io/badge/Numba-00A6FF?style=for-the-badge&logo=numba&logoColor=white
[Numba-url]: https://numba.pydata.org/
[NumPy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[SciPy-shield]: https://img.shields.io/badge/SciPy-8FBC8F?style=for-the-badge&logo=scipy&logoColor=white
[SciPy-url]: https://scipy.org/
