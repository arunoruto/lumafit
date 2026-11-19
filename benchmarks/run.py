"""Pixel-wise fitting benchmark: lumafit vs SciPy (and optimistix, if installed).

Fits every pixel of a synthetic image cube with each solver and writes the
timings to a JSON file, which ``benchmarks/report.py`` turns into the table and
figure shown in the docs.

    uv run --group bench python benchmarks/run.py
    uv run --group bench python benchmarks/report.py
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numba
import numpy as np
import scipy
from numba import jit
from scipy.optimize import least_squares

from lumafit import levenberg_marquardt_pixelwise

OUT = Path(__file__).parent / "results.json"
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
    """Lunar polarization phase curve: p0*sin(a)^p1*cos(a/2)^p2*sin(a-p3)."""
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


def run_optimistix(prob, Y, repeat=3):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optimistix as optx

    t = jnp.asarray(prob["t"])
    if prob is PROBLEMS["decay"]:
        f = lambda p: p[0] * jnp.exp(-t / p[1]) + p[2] * jnp.exp(-t / p[3])  # noqa: E731
    else:
        a = t * jnp.pi / 180.0
        f = lambda p: (  # noqa: E731
            p[0]
            * jnp.sin(a) ** p[1]
            * jnp.cos(a / 2) ** p[2]
            * jnp.sin(a - p[3] * jnp.pi / 180.0)
        )
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    p0 = jnp.asarray(prob["p0"])

    @jax.jit
    def fit_all(Yb):
        def one(y):
            return optx.least_squares(
                lambda p, y: f(p) - y, solver, p0, args=y, max_steps=256, throw=False
            ).value

        return jax.vmap(one)(Yb)

    Yj = jnp.asarray(Y.reshape(-1, M))
    p, best, first = best_of(lambda: fit_all(Yj).block_until_ready(), repeat)
    return np.asarray(p), best, first - best


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

        ref = None
        if not a.only_lumafit:
            ref, s = run_scipy(prob, Y)
            add("scipy least_squares (lm), loop", ref, s)
        for threads in sorted({1, n_threads}):
            p, s, j, it, cv = run_lumafit(prob, Y, threads)
            add(
                f"{a.label}, {threads} thread{'s' * (threads > 1)}",
                p,
                s,
                j,
                it,
                cv,
                ref,
            )
        if prob["jac"] is not None:
            p, s, j, it, cv = run_lumafit(prob, Y, n_threads, prob["jac"])
            add(f"{a.label}, {n_threads} threads, analytic J", p, s, j, it, cv, ref)
        if not a.only_lumafit:
            try:
                p, s, j = run_optimistix(prob, Y)
                add("optimistix LM + jax.vmap (CPU)", p, s, j, ref=ref)
            except ImportError:
                print("optimistix not installed, skipping")

    meta = dict(
        label=a.label,
        pixels=a.side**2,
        samples_per_pixel=M,
        parameters=4,
        threads=n_threads,
        cpu=cpu_name(),
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        scipy=scipy.__version__,
    )
    a.out.write_text(json.dumps(dict(meta=meta, rows=rows), indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
