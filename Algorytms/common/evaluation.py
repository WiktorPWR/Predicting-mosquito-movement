"""Horizon-wise evaluation shared by every prediction algorithm.

A model is any object with
    name: str
    forecast(pos, measured, origins, max_h) -> (len(origins), max_h, 3)
returning, for every origin frame t, the predicted positions at t+1 .. t+max_h
using only frames 0..t.

Protocol
  * test set: the frozen benchmark, Data/Benchmark/ (20 trajectories, common/data.py)
  * origins: every measured (not interpolated) frame after MIN_HISTORY frames
    of history (stride ORIGIN_STRIDE); the history itself is the gap-filled
    sequence, so every model sees the same uniform input
  * horizons: 1 .. MAX_H steps (10 ms .. 5 s)
  * error: Euclidean distance (cm) between predicted and true position; targets
    that are interpolated frames are not scored (only real measurements count)
"""
import json

import numpy as np
import pandas as pd

from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.plots import plot_error_over_time, plot_error_vs_horizon

MAX_H = 500                     # 5 s
MIN_HISTORY = 100               # 1 s of history before the first prediction
ORIGIN_STRIDE = 1
TABLE_STEPS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500]
OVER_TIME_STEPS = [10, 50, 100, 200, 500]   # horizons stored for error-over-time plots / viewer
HIT_CM = (1, 5, 10)
VIEW_ORIGIN_STRIDE = 10         # predictions kept for the 3D viewer: every 0.1 s ...
VIEW_STEP_STRIDE = 5            # ... with every 5th forecast step


def trajectory_errors(model, traj, max_h=MAX_H, min_history=MIN_HISTORY, stride=ORIGIN_STRIDE):
    n = len(traj.pos)
    origins = np.arange(min_history - 1, n - 1, stride)
    origins = origins[traj.measured[origins]]
    preds = model.forecast(traj.pos, traj.measured, origins, max_h)
    idx = origins[:, None] + np.arange(1, max_h + 1)
    valid = idx < n
    idx_c = np.minimum(idx, n - 1)
    valid &= traj.measured[idx_c]
    err = np.linalg.norm(preds - traj.pos[idx_c].astype(np.float32), axis=2)
    err[~valid] = np.nan
    return origins, err, preds


def horizon_stats(err):
    """Per-horizon statistics of an (n_samples, max_h) error array with NaNs."""
    n = (~np.isnan(err)).sum(0)
    with np.errstate(invalid="ignore"):
        st = pd.DataFrame({
            "step": np.arange(1, err.shape[1] + 1),
            "horizon_s": np.arange(1, err.shape[1] + 1) * FRAME_S,
            "n": n,
            "mean_cm": np.nanmean(err, 0),
            "std_cm": np.nanstd(err, 0),
            "rmse_cm": np.sqrt(np.nanmean(err ** 2, 0)),
            "p25_cm": np.nanpercentile(err, 25, axis=0),
            "median_cm": np.nanmedian(err, 0),
            "p75_cm": np.nanpercentile(err, 75, axis=0),
            "p90_cm": np.nanpercentile(err, 90, axis=0),
            "p95_cm": np.nanpercentile(err, 95, axis=0),
            "max_cm": np.nanmax(err, 0),
        })
        for h in HIT_CM:
            st[f"within_{h}cm_%"] = 100 * (err <= h).sum(0) / n
    return st


def score(model, trajs, keep=False):
    """Errors of `model` on every trajectory: (stats over all, per-trajectory rows, raw per-trajectory results)."""
    all_err, per_traj, raw = [], [], []
    for tr in trajs:
        origins, err, preds = trajectory_errors(model, tr)
        all_err.append(err)
        row = {"trajectory_id": tr.tid, "duration_s": round(len(tr.pos) * FRAME_S, 1), "n_origins": len(origins)}
        with np.errstate(invalid="ignore"):
            row.update({f"{s * FRAME_S:g}s": np.nanmean(err[:, s - 1]) for s in TABLE_STEPS})
        per_traj.append(row)
        if keep:
            raw.append((tr, origins, err, preds))
    return horizon_stats(np.concatenate(all_err)), pd.DataFrame(per_traj), raw


def evaluate(model, trajs, extra_params=None, out=None, over_time_figures=False):
    """Score `model` and write the results to `out` (default Results/<model.name>):

    summary.md         table per horizon (mean, median, RMSE, p95, max, hit rates) + per-trajectory means
    error_vs_horizon.png
    horizon_stats.csv  all 500 horizons (used by compare_models)
    params.json
    viewer_data/       for the interactive 3D viewer
    error_over_time/   per-trajectory figures, only with over_time_figures=True
    """
    out = out or RESULTS_DIR / model.name
    label = getattr(model, "label", model.name)
    (out / "viewer_data").mkdir(parents=True, exist_ok=True)
    st, per_traj, raw = score(model, trajs, keep=True)
    steps = np.array(OVER_TIME_STEPS)
    for tr, origins, err, preds in raw:
        vo = np.arange(0, len(origins), VIEW_ORIGIN_STRIDE)
        extra = {}
        if hasattr(model, "cloud"):   # particle-based models: posterior cloud at the viewer's origins
            clouds = model.cloud(tr.pos, tr.measured, origins[vo])
            extra = dict(cloud_states=np.stack([clouds[o][0] for o in origins[vo]]).astype(np.float32),
                         cloud_weights=np.stack([clouds[o][1] for o in origins[vo]]).astype(np.float32))
        np.savez_compressed(out / "viewer_data" / f"traj_{tr.tid:05d}.npz",
                            origins=origins, err_steps=steps, err=err[:, steps - 1],
                            view_origins=origins[vo], view_steps=np.arange(VIEW_STEP_STRIDE, MAX_H + 1, VIEW_STEP_STRIDE),
                            view_preds=preds[vo][:, VIEW_STEP_STRIDE - 1::VIEW_STEP_STRIDE], **extra)
        if over_time_figures:
            (out / "error_over_time").mkdir(exist_ok=True)
            plot_error_over_time(tr, origins, err[:, steps - 1], OVER_TIME_STEPS,
                                 f"{label}: prediction error over flight time, trajectory {tr.tid}",
                                 out / "error_over_time" / f"traj_{tr.tid:05d}.png")
    del raw

    st.to_csv(out / "horizon_stats.csv", index=False)
    plot_error_vs_horizon(st, f"{label}: prediction error vs horizon ({len(trajs)} test trajectories)",
                          out / "error_vs_horizon.png")
    table = st[st["step"].isin(TABLE_STEPS)].reset_index(drop=True)
    pt = per_traj.round(2)
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {label}\n\nPosition error [cm] per prediction horizon, pooled over {len(trajs)} test "
                f"trajectories (every measured frame after {MIN_HISTORY * FRAME_S:g} s of history is a prediction origin).\n\n")
        f.write(markdown_table(table))
        f.write("\n## Mean error [cm] per trajectory\n\n")
        f.write("| " + " | ".join(pt.columns) + " |\n|" + "---|" * len(pt.columns) + "\n")
        f.write("".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in pt.itertuples(index=False)))
    params = dict(getattr(model, "params", {}), **(extra_params or {}), name=model.name, label=label,
                  test_trajectories=[t.tid for t in trajs], max_h=MAX_H, min_history=MIN_HISTORY,
                  origin_stride=ORIGIN_STRIDE)
    (out / "params.json").write_text(json.dumps(params, indent=2, default=float), encoding="utf-8")
    return st


def markdown_table(table):
    cols = ["step", "horizon_s", "n", "mean_cm", "median_cm", "rmse_cm", "p95_cm", "max_cm"] + \
           [f"within_{h}cm_%" for h in HIT_CM]
    t = table[cols].copy()
    t["horizon_s"] = t["horizon_s"].map(lambda v: f"{v:g}")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"
