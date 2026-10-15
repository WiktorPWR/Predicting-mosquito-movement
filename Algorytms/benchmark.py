"""Final benchmark: the selected model of every algorithm, side by side.

Participants (each already evaluated with the common protocol, Algorytms/README.md):
    Kalman filter, particle filter, linear autoregression, GRU, LSTM, transformer
Every model was tuned / selected on training and validation data only; this compares them on the 20 benchmark
trajectories (~72 000 prediction origins per horizon, horizons 10 ms ... 5 s).

Writes Results/benchmark/:
    tables.md                   mean / median / p95 / max error and share within 5 / 10 cm per horizon (best in bold),
                                best model per horizon, model properties
    mean_error_vs_horizon.png   mean error vs horizon (log), one line per model
    median_error_vs_horizon.png
    p95_error_vs_horizon.png
    relative_to_kalman.png      mean error / Kalman filter's mean error
    within_10cm.png             share of predictions within 10 cm vs horizon
The interpretation is in Results/benchmark/README.md (written by hand from these tables).

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.benchmark
"""
import json

import numpy as np
import pandas as pd

from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.evaluation import TABLE_STEPS
from Algorytms.common.plots import INK, INK2, MUTED, SERIES, SURFACE, _fig, _style, plt

OUT = RESULTS_DIR / "benchmark"
# result folder, short name, colour index in SERIES (= index in compare_models.MAIN_MODELS, the viewer's colours)
MODELS = [
    ("kalman/cv_0.336mps", "Kalman filter", 0),
    ("particle/best", "Particle filter", 1),
    ("baselines/linear_ar", "Linear autoregression", 3),
    ("gru/disp_h32_k10", "GRU", 4),
    ("lstm/best", "LSTM", 5),
    ("transformer/best", "Transformer", 6),
]
REF = "Kalman filter"
# what each model needs at run time: parameters and the time of one network call (batch 1, one CPU thread, laptop)
TIMING = {"GRU": ("lstm", "gru_base"), "LSTM": ("lstm", "lstm_b"), "Transformer": ("transformer", "tf_t3")}
PARAMS = {"Kalman filter": "1 tuning value (+ measured noise)", "Particle filter": "4 tuning values, 5 000 particles",
          "Linear autoregression": "183 coefficients"}
METRICS = [("mean_cm", "Mean error [cm]", "{:.2f}"), ("median_cm", "Median error [cm]", "{:.2f}"),
           ("p95_cm", "95th percentile [cm]", "{:.2f}"), ("max_cm", "Maximum error [cm]", "{:.2f}"),
           ("within_5cm_%", "Share within 5 cm [%]", "{:.1f}"), ("within_10cm_%", "Share within 10 cm [%]", "{:.1f}")]
HIGHER_BETTER = ("within_5cm_%", "within_10cm_%")


def load():
    stats, info = {}, {}
    for folder, name, color in MODELS:
        stats[name] = pd.read_csv(RESULTS_DIR / folder / "horizon_stats.csv")
        info[name] = dict(folder=folder, color=SERIES[color],
                          label=json.loads((RESULTS_DIR / folder / "params.json").read_text(encoding="utf-8"))["label"])
    return stats, info


def fmt(v, f):
    return f"{v:.3g}" if v >= 1e4 else f.format(v)


def plot_lines(stats, info, col, ylabel, title, path, log=True, clip=1000, ratio_to=None):
    fig, ax = _fig(figsize=(11, 5.8))
    ref = stats[ratio_to][col].to_numpy() if ratio_to else 1.0
    clipped = False
    for name, st in stats.items():
        y = st[col].to_numpy() / ref
        clipped |= bool(np.nanmax(y) > clip)
        ax.plot(st["horizon_s"], np.minimum(y, clip * 1.5), color=info[name]["color"], lw=2,
                label=name + (" (diverges, clipped)" if np.nanmax(y) > clip else ""))
    ax.set_xscale("log")
    ax.set_xlim(stats[REF]["horizon_s"].min(), stats[REF]["horizon_s"].max())
    if log:
        ax.set_yscale("log")
    if clipped:
        ax.set_ylim(top=clip)
    if ratio_to:      # the reference itself is the line at 1; plain tick labels instead of 10^x
        ticks = [0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3]
        ax.set_ylim(0.4, clip)
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel("prediction horizon [s] (log scale)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def properties(info):
    rows = ["| model | configuration | parameters | network call [ms] |", "|---|---|---|---|"]
    for name in info:
        if name in TIMING:
            study, variant = TIMING[name]
            t = json.loads((RESULTS_DIR / study / "timing_laptop.json").read_text())["variants"][variant]
            p, ms = f"{t['params']:,}".replace(",", " ") + f" ({t['weights_bytes'] / 1024:.1f} kB)", f"{t['call_ms']:.2f}"
        else:
            p, ms = PARAMS[name], "–"
        rows.append(f"| {name} | {info[name]['label']} | {p} | {ms} |")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, info = load()
    cols = [f"{s * FRAME_S:g} s" for s in TABLE_STEPS]
    md = ["# Final benchmark: tables\n",
          "Selected model of every algorithm on the 20 benchmark trajectories; every measured frame after 1 s of "
          f"history is a prediction origin ({int(stats[REF]['n'].iloc[0])} at 10 ms). Position error in cm. "
          "Bold = best model for that horizon. Interpretation: [README.md](README.md).\n",
          "## Participants\n"] + properties(info)
    for col, title, f in METRICS:
        t = pd.DataFrame({n: st.set_index("step").loc[TABLE_STEPS, col].to_numpy() for n, st in stats.items()}, index=cols).T
        t = t.round(1 if col in HIGHER_BETTER else 2)       # ties at the shown precision are all bold
        best = t.max() if col in HIGHER_BETTER else t.min()
        md += [f"\n## {title}\n", "| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for name, r in t.iterrows():
            md.append(f"| {name} | " + " | ".join(f"**{fmt(v, f)}**" if v == best[c] else fmt(v, f) for c, v in r.items()) + " |")
    # winner per horizon over all 500 horizons, as ranges
    mean = pd.DataFrame({n: st["mean_cm"].to_numpy() for n, st in stats.items()}, index=stats[REF]["horizon_s"])
    win, ranges = mean.idxmin(axis=1), []
    for h, w in win.items():
        if ranges and ranges[-1][2] == w:
            ranges[-1][1] = h
        else:
            ranges.append([h, h, w])
    md += ["\n## Best model (lowest mean error) per horizon\n", "| horizons | best model |", "|---|---|"]
    md += [f"| {a:g} – {b:g} s | {w} |" for a, b, w in ranges]
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    plot_lines(stats, info, "mean_cm", "mean position error [cm] (log scale)",
               "Final benchmark: mean error vs prediction horizon", OUT / "mean_error_vs_horizon.png")
    plot_lines(stats, info, "median_cm", "median position error [cm] (log scale)",
               "Final benchmark: median error vs prediction horizon", OUT / "median_error_vs_horizon.png")
    plot_lines(stats, info, "p95_cm", "95th percentile of the error [cm] (log scale)",
               "Final benchmark: 95th percentile of the error vs prediction horizon", OUT / "p95_error_vs_horizon.png")
    plot_lines(stats, info, "mean_cm", "mean error / Kalman filter's mean error (log scale)",
               "Final benchmark: error relative to the Kalman filter (below 1 = better)", OUT / "relative_to_kalman.png",
               clip=3, ratio_to=REF)
    plot_lines(stats, info, "within_10cm_%", "predictions within 10 cm [%]",
               "Final benchmark: share of predictions within 10 cm of the true position", OUT / "within_10cm.png",
               log=False, clip=1e9)
    print("\n".join(md[:9]))
    print(f"\nbest per horizon: " + "; ".join(f"{a:g}-{b:g} s {w}" for a, b, w in ranges))


if __name__ == "__main__":
    main()
