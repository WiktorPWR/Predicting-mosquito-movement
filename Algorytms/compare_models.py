"""Compare every evaluated model: tables and figures across horizons.

Writes Results/comparison/:
    comparison_table.md            mean / median / p95 / max error per model and horizon (best in bold)
    comparison_table.csv
    mean_error_vs_horizon.png, median_error_vs_horizon.png
    relative_to_kalman.png         every model's mean error divided by the Kalman filter's

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.compare_models
"""
import json

import pandas as pd

from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.evaluation import TABLE_STEPS
from Algorytms.common.plots import plot_model_comparison, plot_model_ratio

METRICS = {"mean_cm": "Mean error [cm]", "median_cm": "Median error [cm]", "p95_cm": "95th percentile [cm]",
           "max_cm": "Maximum error [cm]"}


def label(d):
    f = d / "params.json"
    return json.loads(f.read_text(encoding="utf-8")).get("label", d.name) if f.exists() else d.name


def fmt(v):
    return f"{v:.3g}" if v >= 1e4 else f"{v:.2f}"


def main():
    out = RESULTS_DIR / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    stats = {label(p.parent): pd.read_csv(p) for p in sorted(RESULTS_DIR.rglob("horizon_stats.csv"))
             if "comparison" not in p.parts}
    plot_model_comparison(stats, out / "mean_error_vs_horizon.png")
    plot_model_comparison(stats, out / "median_error_vs_horizon.png", col="median_cm", label="median error")
    ref = next((n for n in stats if n.startswith("Kalman")), None)
    if ref:
        plot_model_ratio(stats, ref, out / "relative_to_kalman.png")

    rows, md = [], ["# Model comparison on the benchmark\n",
                    "20 benchmark trajectories, every measured frame after 1 s of history is a prediction origin; "
                    "position error in cm. Bold = best model for that horizon.\n"]
    cols = [f"{s * FRAME_S:g}s" for s in TABLE_STEPS]
    for metric, title in METRICS.items():
        t = pd.DataFrame({name: st.set_index("step").loc[TABLE_STEPS, metric].to_numpy() for name, st in stats.items()},
                         index=cols).T
        best = t.min()
        md += [f"\n## {title}\n", "| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for name, r in t.sort_values("1s").iterrows():
            md.append(f"| {name} | " + " | ".join(f"**{fmt(v)}**" if v == best[c] else fmt(v) for c, v in r.items()) + " |")
            rows.append({"model": name, "metric": metric, **r.to_dict()})
    pd.DataFrame(rows).to_csv(out / "comparison_table.csv", index=False)
    (out / "comparison_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:2 + 3 + len(stats)]))


if __name__ == "__main__":
    main()
