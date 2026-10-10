"""Compare every evaluated model: one figure and one table across horizons.

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.compare_models
"""
import json

import pandas as pd

from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.evaluation import TABLE_STEPS
from Algorytms.common.plots import plot_model_comparison


def main():
    out = RESULTS_DIR / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    def label(d):
        f = d / "params.json"
        return json.loads(f.read_text(encoding="utf-8")).get("label", d.name) if f.exists() else d.name
    stats = {label(p.parent): pd.read_csv(p) for p in sorted(RESULTS_DIR.rglob("horizon_stats.csv"))}
    plot_model_comparison(stats, out / "mean_error_vs_horizon.png")
    plot_model_comparison(stats, out / "median_error_vs_horizon.png", col="median_cm", label="median error")

    rows = []
    for name, st in stats.items():
        t = st.set_index("step").loc[TABLE_STEPS]
        for metric in ("mean_cm", "median_cm", "p95_cm", "max_cm"):
            rows.append({"model": name, "metric": metric, **{f"{s * FRAME_S:g}s": t.at[s, metric] for s in TABLE_STEPS}})
    table = pd.DataFrame(rows).sort_values(["metric", "model"])
    table.to_csv(out / "comparison_table.csv", index=False)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in table.itertuples(index=False)]
    (out / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
