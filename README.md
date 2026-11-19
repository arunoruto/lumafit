<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![PyPI][pypi-shield]][pypi-url]
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/arunoruto/lumafit">
    <img src=".github/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">lumafit</h3>

  <p align="center">
    A Numba-accelerated Levenberg-Marquardt fitting library for Python.
    <br />
    Optimized for pixel-wise fitting on 3D image data.
    <br />
    <a href="https://arunoruto.github.io/lumafit/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/arunoruto/lumafit/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/arunoruto/lumafit/issues/new?labels=enhancement">Request Feature</a>
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
    <li><a href="#performance">Performance</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

![Time per fit for lumafit, SciPy and optimistix on CPU and GPU][benchmark-light]
![Time per fit for lumafit, SciPy and optimistix on CPU and GPU][benchmark-dark]

`lumafit` fits many small non-linear least squares problems at once, such as
one curve along the third dimension for every pixel of an image cube. Calling
`scipy.optimize.least_squares` in a Python loop spends most of its time in the
interpreter; `lumafit` implements the Levenberg-Marquardt algorithm in
[Numba](https://numba.pydata.org/), compiles it together with your model and
runs the fits on all cores, or on a GPU.

- `fit_curve` fits a single curve.
- `fit_cube` fits every pixel of a `(rows, cols, samples)` cube in parallel and
  skips pixels masked with NaN.
- Weighted least squares, returning the covariance matrix of the parameters.
- Finite difference Jacobian by default, analytical Jacobian if you have one.
- `fit_curve` is a nopython function, so it can be called from your own
  Numba-compiled code.
- `fit_cube_pointwise` runs the same fit for a pointwise model, on the CPU or,
  through [numba-vulkan](https://github.com/arunoruto/numba-vulkan), on any GPU
  with a Vulkan driver.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][Python-shield]][Python-url]
- [![Numba][Numba-shield]][Numba-url]
- [![NumPy][NumPy-shield]][NumPy-url]
- [![SciPy][SciPy-shield]][SciPy-url]
- [![Vulkan][Vulkan-shield]][Vulkan-url] (optional, through numba-vulkan)
- [![uv][uv-shield]][uv-url]
- [![Nix][Nix-shield]][Nix-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

Python 3.10 or newer. NumPy, Numba and SciPy are installed as dependencies;
SciPy provides the BLAS/LAPACK routines Numba uses for linear algebra.

### Installation

From [PyPI][pypi-url]:

```sh
pip install lumafit
# or
uv add lumafit
```

With GPU support (Python 3.11 or newer, a Vulkan loader and a driver for your
GPU):

```sh
pip install "lumafit[vulkan]"
```

From source:

```sh
git clone https://github.com/arunoruto/lumafit.git
cd lumafit
pip install .
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

The model is a Numba nopython function. It takes the parameters first,
followed by whatever you pass as `args` to the solver, typically the
independent variable.

```python
import numpy as np
from numba import njit
from lumafit import fit_curve, fit_cube


@njit
def decay(p, t):
    return p[0] * np.exp(-t / p[1]) + p[2] * np.exp(-t / p[3])


t = np.linspace(0.1, 25, 100)
p_true = np.array([5.0, 2.0, 2.0, 10.0])
p0 = np.array([4.0, 1.5, 1.5, 8.0])
rng = np.random.default_rng(0)
```

### Fitting a single curve

```python
y = decay(p_true, t) + rng.normal(0, 0.02, t.shape)

result = fit_curve(decay, p0, y, args=(t,))
result.p, result.cov, result.chi2, result.n_iter, result.converged
```

### Fitting pixel-wise on 3D data

```python
cube = decay(p_true, t) + rng.normal(0, 0.02, (128, 128, t.size))
cube[0, 0, :] = np.nan  # masked pixels are skipped

result = fit_cube(decay, p0, cube, args=(t,))

result.p.shape           # (128, 128, 4)
result.converged.mean()  # fraction of pixels that converged

p, cov, chi2, n_iter, converged = result  # a named tuple, so it also unpacks
```

The first call compiles the solver for your model, which takes around ten
seconds; after that the 16,384 fits above run in about 0.15 s on 16 threads.

### Fitting on a GPU

`fit_cube_pointwise` takes a model that computes one sample at a time. It
compiles in about two seconds and runs on the CPU threads or on a Vulkan
device:

```python
import math
from lumafit import fit_cube_pointwise


@njit
def decay_point(p, t):
    return p[0] * math.exp(-t / p[1]) + p[2] * math.exp(-t / p[3])


result = fit_cube_pointwise(
    decay_point, p0, cube, t, target="vulkan", dtype=np.float32
)
```

On an NVIDIA TITAN X the same 16,384 fits take 9 ms in `float32`, where the
results agree with `float64` to about 1e-4. Leave out `target` to run it on the
CPU.

Weights (`1 / sigma**2`), an analytical Jacobian (`jac`) and minimizing a
residual function directly are covered in the
[Documentation](https://arunoruto.github.io/lumafit/usage.html).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PERFORMANCE -->

## Performance

Time per fit for a 128 × 128 cube with 100 samples per pixel and 4 parameters,
on an Intel Core i9-9900K (8 cores, 16 threads) with an NVIDIA TITAN X and the
CPU's integrated Intel UHD 630, NumPy arrays in and out:

| Solver                                           | Exponential decay | Polarization curve |
| ------------------------------------------------ | ----------------: | -----------------: |
| `scipy.optimize.least_squares`, loop             |            840 µs |             984 µs |
| `optimistix` + `jax.vmap`, CPU                   |            205 µs |             451 µs |
| `optimistix` + `jax.vmap`, CUDA on TITAN X       |           26.4 µs |            56.5 µs |
| `optimistix` + `jax.vmap`, CUDA on TITAN X, f32  |            9.5 µs |            16.6 µs |
| lumafit 0.2.3, 16 threads                        |           15.4 µs |            32.3 µs |
| **`fit_cube`, 16 threads**                       |            8.4 µs |            21.9 µs |
| **`fit_cube_pointwise`, 16 threads**             |            7.0 µs |            22.3 µs |
| **`fit_cube_pointwise`, 16 threads, f32**        |            3.1 µs |             9.5 µs |
| **`fit_cube_pointwise`, Vulkan on UHD 630, f32** |            3.4 µs |            14.5 µs |
| **`fit_cube_pointwise`, Vulkan on TITAN X**      |            2.8 µs |            19.7 µs |
| **`fit_cube_pointwise`, Vulkan on TITAN X, f32** |           0.51 µs |             1.1 µs |

All solvers find the same minimum: the fitted parameters agree to 1e-7 or better
in double precision and to about 1e-4 in single precision (f32). The one-off
compilation (about ten seconds for `fit_cube`, two for `fit_cube_pointwise`) is
not included, so for a few hundred curves SciPy is still the quicker choice.
Methodology and the full table are in the
[benchmark docs](https://arunoruto.github.io/lumafit/benchmarks.html); to
reproduce:

```sh
uv run --group bench python benchmarks/run.py
uv run --group bench python benchmarks/report.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEVELOPMENT -->

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --group test --group docs
uv run pytest                                              # tests and doctests
uv run sphinx-build -M html docs/source docs/build -W      # documentation
```

The tests check the solver against known solutions and against
`scipy.optimize.least_squares`.

On Nix, the flake provides the same environment (loaded automatically with
direnv):

```sh
nix develop .#uv2nix
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [x] Support analytical Jacobian functions.
- [x] Nielsen's damping strategy.
- [x] Documentation and examples.
- [x] Publish on PyPI.
- [x] Run on GPUs, through Vulkan.
- [ ] Implement parameter bounds.
- [ ] Improve robustness for ill-conditioned problems.

See the [open issues](https://github.com/arunoruto/lumafit/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are welcome! If you have suggestions or find bugs, please open an issue or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Releases are cut with
[python-semantic-release](https://python-semantic-release.readthedocs.io/), so
commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the [MIT License][license-url]. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Mirza Arnaut - mirza.arnaut@tu-dortmund.de

Project Link: [https://github.com/arunoruto/lumafit](https://github.com/arunoruto/lumafit)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- Levenberg (1944) and Marquardt (1963) for the algorithm, Nielsen (1999) for
  the damping update and Gavin (2020) for a very readable description; full
  [references](https://arunoruto.github.io/lumafit/references.html) are in the docs.
- [Numba](https://numba.pydata.org/) for providing the acceleration capabilities.
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for fundamental numerical computing tools.
- [pytest](https://docs.pytest.org/) for the testing framework.
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template) for the README structure template.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[benchmark-light]: docs/source/benchmarks/benchmark-light.svg#gh-light-mode-only
[benchmark-dark]: docs/source/benchmarks/benchmark-dark.svg#gh-dark-mode-only
[pypi-shield]: https://img.shields.io/pypi/v/lumafit.svg?style=for-the-badge
[pypi-url]: https://pypi.org/project/lumafit/
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
[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Numba-shield]: https://img.shields.io/badge/Numba-00A3E0?style=for-the-badge&logo=numba&logoColor=white
[Numba-url]: https://numba.pydata.org/
[NumPy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[SciPy-shield]: https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white
[SciPy-url]: https://scipy.org/
[Vulkan-shield]: https://img.shields.io/badge/Vulkan-AC162C?style=for-the-badge&logo=vulkan&logoColor=white
[Vulkan-url]: https://github.com/arunoruto/numba-vulkan
[uv-shield]: https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white
[uv-url]: https://docs.astral.sh/uv/
[Nix-shield]: https://img.shields.io/badge/Nix-5277C3?style=for-the-badge&logo=nixos&logoColor=white
[Nix-url]: https://nixos.org/
