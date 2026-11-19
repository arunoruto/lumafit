"""Turn ``benchmarks/results*.json`` into the table and figures used in the docs.

uv run --group bench python benchmarks/report.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DOCS = HERE.parent / "docs" / "source"
SCIPY = "SciPy least_squares, Python loop"
PACKAGES = {
    "python": "Python",
    "numpy": "NumPy",
    "numba": "Numba",
    "scipy": "SciPy",
    "numba-vulkan": "numba-vulkan",
    "jax": "JAX",
    "optimistix": "optimistix",
}
TITLES = {
    "decay": "Double exponential decay",
    "polarization": "Polarization phase curve",
}
# Emphasis palette: the current lumafit in the accent hue, everything else gray.
# Each figure carries its own surface so it stays readable on any page background.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        text="#0b0b0b",
        muted="#52514e",
        accent="#2a78d6",
        other="#898781",
        grid="#e1e0d9",
    ),
    "dark": dict(
        surface="#1a1a19",
        text="#f0efec",
        muted="#c3c2b7",
        accent="#3987e5",
        other="#898781",
        grid="#2c2c2a",
    ),
}


def load():
    """Merge all result files; the default ``results.json`` is the current run."""
    current = json.loads((HERE / "results.json").read_text())
    rows = list(current["rows"])
    for path in sorted(HERE.glob("results-*.json")):
        rows += json.loads(path.read_text())["rows"]
    return current["meta"], rows


def by_problem(rows):
    out = {}
    for r in rows:
        out.setdefault(r["problem"], []).append(r)
    return {k: sorted(v, key=lambda r: -r["us_per_fit"]) for k, v in out.items()}


def fmt(value, spec, missing="-"):
    return missing if value is None else format(value, spec)


def table(meta, problems):
    gpus = f" with {' and '.join(meta['gpus'])}" if meta.get("gpus") else ""
    lines = [
        f"{meta['pixels']:,} fits per run ({meta['samples_per_pixel']} samples, "
        f"{meta['parameters']} parameters each) on an {meta['cpu']} "
        f"({meta['threads']} threads){gpus}; "
        + ", ".join(
            f"{name} {meta[key]}" for key, name in PACKAGES.items() if key in meta
        )
        + ".",
    ]
    for name, rows in problems.items():
        scipy = next(r["us_per_fit"] for r in rows if r["solver"] == SCIPY)
        lines += [
            "",
            f"### {TITLES[name]}",
            "",
            "| Solver | Total | Per fit | vs. SciPy | Iterations | First-call JIT | Median deviation from SciPy |",
            "|---|--:|--:|--:|--:|--:|--:|",
        ]
        for r in rows:
            bold = "**" if r["solver"].startswith(meta["label"]) else ""
            # Only the first run of a model in a process pays for compilation
            jit = f"{r['jit_seconds']:.1f} s" if (r["jit_seconds"] or 0) > 0.1 else "-"
            lines.append(
                f"| {bold}{r['solver']}{bold} | {r['seconds']:.3f} s | {micros(r['us_per_fit'])} µs "
                f"| {scipy / r['us_per_fit']:.0f}× | {fmt(r['mean_iterations'], '.1f')} "
                f"| {jit} | {fmt(r['median_rel_diff_vs_scipy'], '.0e')} |"
            )
    return "\n".join(lines) + "\n"


def micros(value):
    """Three significant figures at most: 0.53, 8.4, 206."""
    return (
        f"{value:.2f}"
        if value < 1
        else f"{value:.1f}"
        if value < 100
        else f"{value:.0f}"
    )


def figure(meta, problems, theme, path):
    """Dot plot on a log axis: the times span three orders of magnitude."""
    c = THEMES[theme]
    n_rows = [len(v) for v in problems.values()]
    fig, axes = plt.subplots(
        len(problems), 1, figsize=(9.5, 0.3 * sum(n_rows) + 1.8), height_ratios=n_rows
    )
    for ax, (name, rows) in zip(axes, problems.items(), strict=True):
        scipy = next(r["us_per_fit"] for r in rows if r["solver"] == SCIPY)
        y = range(len(rows))[::-1]  # slowest on top
        for yi, r in zip(y, rows, strict=True):
            ours = r["solver"].startswith(meta["label"])
            ax.plot(
                r["us_per_fit"], yi, "o", markersize=7,
                color=c["accent"] if ours else c["other"],
            )  # fmt: skip
            label = f"{micros(r['us_per_fit'])} µs"
            if r["solver"] != SCIPY:
                label += f"  ({scipy / r['us_per_fit']:.0f}×)"
            ax.annotate(
                label, (r["us_per_fit"], yi), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=c["text"],
            )  # fmt: skip
        names = [r["solver"].replace(meta["label"], "lumafit") for r in rows]
        ax.set_yticks(list(y), names, fontsize=8.5, color=c["text"])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 6000)
        ax.set_xticks([1, 10, 100, 1000], ["1 µs", "10 µs", "100 µs", "1 ms"])
        ax.minorticks_off()
        ax.grid(axis="both", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor(c["surface"])
        ax.set_title(
            TITLES[name], loc="left", fontsize=11, color=c["text"], fontweight="bold"
        )
        ax.tick_params(axis="x", colors=c["muted"], labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel(
        "time per fit, log scale (lower is better; in brackets: speed-up over SciPy)",
        fontsize=9, color=c["muted"],
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, facecolor=c["surface"])
    plt.close(fig)


def main():
    meta, rows = load()
    problems = by_problem(rows)
    out = DOCS / "benchmarks"
    out.mkdir(exist_ok=True)
    (out / "results.md").write_text(table(meta, problems))
    for theme in THEMES:
        figure(meta, problems, theme, out / f"benchmark-{theme}.svg")
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
