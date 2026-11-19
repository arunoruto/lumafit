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
SCIPY = "scipy least_squares (lm), loop"
TITLES = {
    "decay": "Double exponential decay",
    "polarization": "Polarization phase curve",
}
# Emphasis palette: the current lumafit in the accent hue, everything else gray.
THEMES = {
    "light": dict(text="#0b0b0b", muted="#52514e", accent="#2a78d6", other="#c3c2b7"),
    "dark": dict(text="#f0efec", muted="#c3c2b7", accent="#3987e5", other="#52514e"),
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
    lines = [
        f"{meta['pixels']:,} fits per run ({meta['samples_per_pixel']} samples, "
        f"{meta['parameters']} parameters each) on an {meta['cpu']} "
        f"({meta['threads']} threads); Python {meta['python']}, NumPy {meta['numpy']}, "
        f"Numba {meta['numba']}, SciPy {meta['scipy']}.",
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
                f"| {bold}{r['solver']}{bold} | {r['seconds']:.3f} s | {r['us_per_fit']:.1f} µs "
                f"| {scipy / r['us_per_fit']:.1f}× | {fmt(r['mean_iterations'], '.1f')} "
                f"| {jit} | {fmt(r['median_rel_diff_vs_scipy'], '.0e')} |"
            )
    return "\n".join(lines) + "\n"


def figure(meta, problems, theme, path):
    c = THEMES[theme]
    n_rows = [len(v) for v in problems.values()]
    fig, axes = plt.subplots(
        len(problems), 1, figsize=(8, 0.42 * sum(n_rows) + 1.6), height_ratios=n_rows
    )
    for ax, (name, rows) in zip(axes, problems.items(), strict=True):
        scipy = next(r["us_per_fit"] for r in rows if r["solver"] == SCIPY)
        values = [r["us_per_fit"] for r in rows]
        colors = [
            c["accent"] if r["solver"].startswith(meta["label"]) else c["other"]
            for r in rows
        ]
        y = range(len(rows))[::-1]  # slowest on top
        ax.barh(y, values, height=0.55, color=colors)
        for yi, r in zip(y, rows, strict=True):
            label = (
                f"{r['us_per_fit']:.0f} µs"
                if r["us_per_fit"] >= 100
                else f"{r['us_per_fit']:.1f} µs"
            )
            if r["solver"] != SCIPY:
                label += f"  ({scipy / r['us_per_fit']:.0f}× faster than SciPy)"
            ax.text(
                r["us_per_fit"] + max(values) * 0.012,
                yi,
                label,
                va="center",
                fontsize=9,
                color=c["text"],
            )
        ax.set_yticks(list(y), [r["solver"] for r in rows], fontsize=9, color=c["text"])
        ax.set_title(
            TITLES[name], loc="left", fontsize=11, color=c["text"], fontweight="bold"
        )
        ax.set_xlim(0, max(values) * 1.45)
        ax.tick_params(axis="x", colors=c["muted"], labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(c["muted"])
    axes[-1].set_xlabel(
        "time per fit in µs (lower is better)", fontsize=9, color=c["muted"]
    )
    fig.tight_layout()
    fig.savefig(path, transparent=True)
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
