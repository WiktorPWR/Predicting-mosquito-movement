"""Sweep the Kalman CV velocity setting.

N_SETTINGS values evenly spaced over (0, v_max]: v_max * k / N for k = N..1.
Zero itself is left out: with no process noise and no velocity uncertainty the
gain decays to zero and the filter stops following the measurements.

Writes only the comparison to Results/kalman/sweep/:
    sweep_mean_error.md                table: mean error per setting at key horizons (+ best marked)
    sweep_mean_error_by_horizon.csv    mean error for all 500 horizons x settings
    sweep_mean_error.png, sweep_relative_error.png

The chosen setting is fixed in kalman_filter.SETTING_CM_S; evaluate it with
    .venv/Scripts/python -m Algorytms.kalman.run

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.kalman.sweep            # run the sweep (~7 min)
    .venv/Scripts/python -m Algorytms.kalman.sweep --plot     # redraw from the saved CSV
"""
import sys

import numpy as np
import pandas as pd

from Algorytms.common import data
from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.evaluation import TABLE_STEPS, score
from Algorytms.common.plots import plot_sweep
from Algorytms.kalman.kalman_filter import SETTING_CM_S, KalmanCV

N_SETTINGS = 10
SWEEP_DIR = RESULTS_DIR / "kalman" / "sweep"
CSV = SWEEP_DIR / "sweep_mean_error_by_horizon.csv"


def write_table(df):
    """Markdown table of mean error at TABLE_STEPS, one row per setting; best per horizon in bold."""
    t = df.set_index("horizon_s").iloc[np.array(TABLE_STEPS) - 1].T
    t.columns = [f"{h:g}s" for h in t.columns]
    rel = (t / t.min()).mean(axis=1)
    lines = ["| setting [m/s] | " + " | ".join(t.columns) + " | mean error / best |", "|" + "---|" * (len(t.columns) + 2)]
    for s, r in t.iterrows():
        cells = [f"**{v:.2f}**" if v == t[c].min() else f"{v:.2f}" for c, v in r.items()]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {rel[s]:.4f} |")
    (SWEEP_DIR / "sweep_mean_error.md").write_text(
        "# Kalman CV: velocity setting sweep\n\nMean position error [cm] per prediction horizon on the test set; "
        "bold = best setting for that horizon. Last column: mean ratio to the best setting over these horizons "
        "(lower is better).\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return t, rel


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    if "--plot" not in sys.argv:
        train = data.train_set()
        v_max = data.max_speed(train)
        meas_std = data.measurement_noise(train)
        settings = v_max * np.arange(1, N_SETTINGS + 1) / N_SETTINGS
        print(f"v_max = {v_max / 100:.3f} m/s, settings [m/s]: {np.round(settings / 100, 3)}")
        test = data.test_set()
        df = pd.DataFrame({"horizon_s": np.arange(1, 501) * FRAME_S})
        for s in settings:
            df[f"{s / 100:.3f}"] = score(KalmanCV(meas_std, setting=s), test)[0]["mean_cm"].to_numpy()
            print(f"  setting {s / 100:.3f} m/s done")
        df.to_csv(CSV, index=False)
    df = pd.read_csv(CSV, dtype={"horizon_s": float})
    df.columns = ["horizon_s"] + [f"{float(c):.3f}" for c in df.columns[1:]]
    t, rel = write_table(df)
    plot_sweep(df, SWEEP_DIR, chosen=f"{SETTING_CM_S / 100:.3f}")
    print(t.round(2).to_string())
    print(f"lowest mean error / best: {rel.idxmin()} m/s")


if __name__ == "__main__":
    main()
