"""Pixel-wise fitting benchmark: lumafit vs SciPy and optimistix, on CPU and GPU.

Fits every pixel of a synthetic image cube with each solver and writes the
timings to a JSON file, which ``benchmarks/report.py`` turns into the table and
figure shown in the docs. Every timing is end to end: NumPy arrays in, NumPy
arrays out. Solvers whose packages or devices are missing are skipped.

    uv run --group bench python benchmarks/run.py
    uv run --group bench python benchmarks/report.py

For the JAX GPU rows, add ``--with "jax[cuda12]"`` to the first command.
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np
import scipy
from numba import jit, njit
from scipy.optimize import least_squares

from lumafit import levenberg_marquardt_pixelwise, make_pixelwise_fitter

OUT = Path(__file__).parent / "results.json"
SCIPY = "SciPy least_squares, Python loop"
M = 100  # samples per pixel


# --- models (same two as the test suite) ------------------------------------
@jit(nopython=True, cache=True)
def decay(p, *args):
    """Double exponential decay: p0*exp(-t/p1) + p2*exp(-t/p3)."""
    t = args[0]
    return p[0] * np.exp(-t / p[1]) + p[2] * np.exp(-t / p[3])


@jit(nopython=True, cache=True)
def decay_jac(p, *args):
    t = args[0]
    J = np.empty((t.shape[0], 4))
    e1 = np.exp(-t / p[1])
    e2 = np.exp(-t / p[3])
    J[:, 0] = e1
    J[:, 1] = p[0] * e1 * t / p[1] ** 2
    J[:, 2] = e2
    J[:, 3] = p[2] * e2 * t / p[3] ** 2
    return J


@jit(nopython=True, cache=True)
def polarization(p, *args):
    """Polarization phase curve of Lumme & Muinonen (1993): p0*sin(a)^p1*cos(a/2)^p2*sin(a-p3)."""
    a = args[0] * np.pi / 180.0
    return (
        p[0]
        * np.sin(a) ** p[1]
        * np.cos(a / 2) ** p[2]
        * np.sin(a - p[3] * np.pi / 180.0)
    )


PROBLEMS = {
    "decay": dict(
        model=decay,
        jac=decay_jac,
        t=np.linspace(0.1, 25, M),
        p_true=np.array([5.0, 2.0, 2.0, 10.0]),
        p0=np.array([4.0, 1.5, 1.5, 8.0]),
        noise=0.02,
    ),
    "polarization": dict(
        model=polarization,
        jac=None,
        t=np.linspace(1.0, 89.0, M),
        p_true=np.array([2.0, 3.0, 5.0, 18.0]),
        p0=np.array([1.8, 2.7, 4.5, 16.0]),
        noise=0.005,
    ),
}


def pointwise_model(name, F):
    """The same models, one sample at a time, with constants typed as `F`."""
    DEG, HALF = F(np.pi / 180), F(0.5)

    @njit
    def decay_pt(p, t):
        return p[0] * math.exp(-t / p[1]) + p[2] * math.exp(-t / p[3])

    @njit
    def polarization_pt(p, t):
        a = t * DEG
        return (
            p[0]
            * math.sin(a) ** p[1]
            * math.cos(a * HALF) ** p[2]
            * math.sin(a - p[3] * DEG)
        )

    return {"decay": decay_pt, "polarization": polarization_pt}[name]


def make_cube(prob, side, seed=0):
    """Image cube where every pixel has its own true parameters (+-10 %)."""
    rng = np.random.default_rng(seed)
    p = prob["p_true"] * rng.uniform(0.9, 1.1, (side, side, 4))
    y = np.empty((side, side, M))
    for r in range(side):
        for c in range(side):
            y[r, c] = prob["model"](p[r, c], prob["t"])
    return y + rng.normal(0, prob["noise"], y.shape)


def best_of(fn, repeat):
    """(result, best wall time, first-call wall time)."""
    t0 = time.perf_counter()
    out = fn()
    first = time.perf_counter() - t0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, min(times), first


def run_scipy(prob, Y):
    model, t = prob["model"].py_func, prob["t"]  # plain NumPy, no JIT dispatch cost
    flat = Y.reshape(-1, M)
    p = np.empty((flat.shape[0], 4))
    t0 = time.perf_counter()
    for i, y in enumerate(flat):
        p[i] = least_squares(lambda q, y=y: model(q, t) - y, prob["p0"], method="lm").x
    return p, time.perf_counter() - t0


def run_lumafit(prob, Y, threads, jac=None, repeat=3):
    numba.set_num_threads(threads)
    fit = lambda: levenberg_marquardt_pixelwise(  # noqa: E731
        prob["model"], prob["p0"], Y, jac_func=jac, args_for_each_pixel=(prob["t"],)
    )
    (p, _, _, n_iter, conv), best, first = best_of(fit, repeat)
    return (
        p.reshape(-1, 4),
        best,
        first - best,
        float(n_iter.mean()),
        float(conv.mean()),
    )


def run_pointwise(name, prob, Y, dtype, target, device=None, threads=None, repeat=3):
    if threads:
        numba.set_num_threads(threads)
    fit = make_pixelwise_fitter(pointwise_model(name, dtype), 4, dtype, target)
    Y = Y.astype(dtype)  # like the other solvers, timed on arrays of its own dtype
    (p, _, _, n_iter, conv), best, first = best_of(
        lambda: fit(prob["p0"], Y, prob["t"], device=device), repeat
    )
    return (
        p.reshape(-1, 4),
        best,
        first - best,
        float(n_iter.mean()),
        float(conv.mean()),
    )


def run_optimistix(prob, Y, dtype, device, repeat=3):
    import jax

    jax.config.update("jax_enable_x64", dtype == np.float64)
    import jax.numpy as jnp
    import optimistix as optx

    tol = 1e-8 if dtype == np.float64 else math.sqrt(np.finfo(dtype).eps)
    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    Y = Y.reshape(-1, M).astype(dtype)
    with jax.default_device(device):
        t = jnp.asarray(prob["t"], dtype)
        p0 = jnp.asarray(prob["p0"], dtype)
        if prob is PROBLEMS["decay"]:
            f = lambda p: p[0] * jnp.exp(-t / p[1]) + p[2] * jnp.exp(-t / p[3])  # noqa: E731
        else:
            a = t * (jnp.pi / 180.0)
            f = lambda p: (  # noqa: E731
                p[0] * jnp.sin(a) ** p[1] * jnp.cos(a / 2) ** p[2]
                * jnp.sin(a - p[3] * (jnp.pi / 180.0))
            )  # fmt: skip

        @jax.jit
        def fit_all(Yb):
            def one(y):
                return optx.least_squares(
                    lambda p, y: f(p) - y,
                    solver,
                    p0,
                    args=y,
                    max_steps=256,
                    throw=False,
                ).value

            return jax.vmap(one)(Yb)

        p, best, first = best_of(lambda: np.asarray(fit_all(jnp.asarray(Y))), repeat)
    return p, best, first - best


def short(device_name):
    """'Intel(R) UHD Graphics 630 (CFL GT2)' -> 'Intel UHD Graphics 630'."""
    return device_name.replace("(R)", "").split(" (")[0].strip()


def versions(*packages):
    """Installed versions of the optional packages, for the report."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            pass
    return out


def cpu_name():
    try:
        info = Path("/proc/cpuinfo").read_text()
        return next(
            ln.split(":", 1)[1].strip()
            for ln in info.splitlines()
            if "model name" in ln
        )
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser(description="Pixel-wise fitting benchmark")
    ap.add_argument("--side", type=int, default=128, help="image is side x side pixels")
    ap.add_argument("--label", default="lumafit (this version)")
    ap.add_argument(
        "--only-lumafit", action="store_true", help="skip SciPy and optimistix"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    n_threads = numba.get_num_threads()
    rows = []
    gpus = set()
    for name, prob in PROBLEMS.items():
        Y = make_cube(prob, a.side)
        n_px = a.side**2

        def add(
            solver,
            p,
            seconds,
            jit_s=None,
            iters=None,
            conv=None,
            ref=None,
            name=name,
            n_px=n_px,
        ):
            err = (
                None
                if ref is None
                else float(np.nanmedian(np.abs(p - ref) / np.abs(ref)))
            )
            row = dict(
                problem=name,
                solver=solver,
                seconds=seconds,
                us_per_fit=seconds / n_px * 1e6,
                jit_seconds=jit_s,
                mean_iterations=iters,
                converged=conv,
                median_rel_diff_vs_scipy=err,
            )
            rows.append(row)
            print(
                "  ".join(
                    f"{v:.3g}" if isinstance(v, float) else str(v) for v in row.values()
                )
            )

        def timed(solver, run, *args, **kwargs):
            """Run one solver and record it; a missing package or device skips it."""
            try:
                out = run(*args, **kwargs)
            except ImportError as exc:
                print(f"skipping {solver}: {exc}")
                return None
            add(solver, *out, ref=ref)  # noqa: B023 (ref is read at call time on purpose)
            return out[0]

        ref = None
        if not a.only_lumafit:
            ref = timed(SCIPY, run_scipy, prob, Y)
        for threads in sorted({1, n_threads}):
            unit = f"{threads} thread{'s' * (threads > 1)}"
            timed(f"{a.label}, array model, {unit}", run_lumafit, prob, Y, threads)
        if prob["jac"] is not None:
            timed(
                f"{a.label}, array model, {n_threads} threads, analytic J",
                run_lumafit, prob, Y, n_threads, prob["jac"],
            )  # fmt: skip
        if a.only_lumafit:
            continue

        pointwise = f"{a.label}, pointwise"
        for dtype, suffix in ((np.float64, ""), (np.float32, ", float32")):
            timed(
                f"{pointwise}, {n_threads} threads{suffix}",
                run_pointwise, name, prob, Y, dtype, "cpu", threads=n_threads,
            )  # fmt: skip
        try:
            import numba_vulkan as nv

            vulkan_devices = [
                d for d in nv.list_devices() if d.kind in ("discrete", "integrated")
            ]
        except ImportError as exc:
            print(f"skipping Vulkan: {exc}")
            vulkan_devices = []
        for dev in vulkan_devices:
            gpus.add(short(dev.name))
            if dev.kind == "discrete" and dev.float64:
                timed(
                    f"{pointwise}, Vulkan on {short(dev.name)}",
                    run_pointwise, name, prob, Y, np.float64, "vulkan", dev.index,
                )  # fmt: skip
            timed(
                f"{pointwise}, Vulkan on {short(dev.name)}, float32",
                run_pointwise, name, prob, Y, np.float32, "vulkan", dev.index,
            )  # fmt: skip

        try:
            import jax

            jax_devices = [("CPU", jax.devices("cpu")[0])]
            try:
                gpu = jax.devices("gpu")[0]
                jax_devices.append((f"CUDA on {short(gpu.device_kind)}", gpu))
            except RuntimeError:
                print("skipping JAX on the GPU: no CUDA-enabled jaxlib")
        except ImportError as exc:
            print(f"skipping optimistix: {exc}")
            jax_devices = []
        for where, device in jax_devices:
            timed(
                f"optimistix + jax.vmap, {where}",
                run_optimistix,
                prob,
                Y,
                np.float64,
                device,
            )
            if where != "CPU":
                timed(
                    f"optimistix + jax.vmap, {where}, float32",
                    run_optimistix, prob, Y, np.float32, device,
                )  # fmt: skip

    meta = dict(
        label=a.label,
        gpus=sorted(gpus),
        pixels=a.side**2,
        samples_per_pixel=M,
        parameters=4,
        threads=n_threads,
        cpu=cpu_name(),
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        scipy=scipy.__version__,
        **versions("jax", "optimistix", "numba-vulkan"),
    )
    a.out.write_text(json.dumps(dict(meta=meta, rows=rows), indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
