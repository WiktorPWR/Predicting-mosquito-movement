"""Full evaluation of the best particle-filter configuration + convergence analysis.

Reads the configuration chosen by the sweeps (Results/particle/sweep/best.json) and writes
    Results/particle/best/          tables, error-vs-horizon figure, viewer data with particle clouds
    Results/particle/convergence/   from which sample on predictions are sensible (PF vs Kalman)

Usage (from the repo root, after `python -m Algorytms.particle.sweep grid`):
    .venv/Scripts/python -m Algorytms.particle.run            # ~20 min
    .venv/Scripts/python -m Algorytms.particle.run report     # redraw the convergence table/figure only
"""
import json
import sys

import pandas as pd

from Algorytms.common import data
from Algorytms.common.convergence import HORIZONS, TOLERANCE, cold_start_errors, sensible_from
from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.evaluation import evaluate
from Algorytms.common.plots import plot_convergence
from Algorytms.kalman.kalman_filter import KalmanCV
from Algorytms.particle.sweep import SWEEP_DIR, make_filter

OUT = RESULTS_DIR / "particle"


def convergence(models, trajs):
    out = OUT / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    curves = {}
    for m in models:
        curves[m.label] = cold_start_errors(m, trajs)
        print(f"  convergence {m.label}: {curves[m.label].attrs['n_starts']} cold starts")
    pd.concat({lab: df.set_index("k") for lab, df in curves.items()}, axis=1).to_csv(out / "convergence_by_sample.csv")
    (out / "n_starts.txt").write_text(str(next(iter(curves.values())).attrs["n_starts"]))
    convergence_report()


def convergence_report():
    """Table + figure from the saved cold-start curves (also callable alone: `run.py report`)."""
    out = OUT / "convergence"
    wide = pd.read_csv(out / "convergence_by_sample.csv", header=[0, 1], index_col=0)
    n_starts = (out / "n_starts.txt").read_text().strip()
    curves = {lab: wide[lab].reset_index() for lab in wide.columns.get_level_values(0).unique()}
    first = {lab: sensible_from(df) for lab, df in curves.items()}
    plot_convergence(curves, first, out / "convergence.png", "Error after a cold start")
    lines = [f"# From which sample on are predictions sensible?\n",
             f"Every model was started from scratch {n_starts} times (every 1 s of every test trajectory) and fed "
             f"only the frames from that point on. *Sensible from sample k* = first k after which the mean error "
             f"stays within {TOLERANCE:.0%} of its steady-state value (mean over samples 201-300).\n",
             "| model | horizon | sensible from sample | = time after start | error at sample 1 [cm] | steady-state error [cm] |",
             "|---|---|---|---|---|---|"]
    for lab, f in first.items():
        for h in HORIZONS:
            r = f[h]
            lines.append(f"| {lab} | {h * FRAME_S:g} s | {r['first_k']} | {r['first_k'] * FRAME_S:.2f} s | "
                         f"{r['error_at_k1_cm']:.2f} | {r['steady_cm']:.2f} |")
    (out / "convergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[3:]))


def main():
    if "report" in sys.argv:
        return convergence_report()
    cfg = json.loads((SWEEP_DIR / "best.json").read_text(encoding="utf-8"))
    pf = make_filter(cfg["setting"], cfg["like_sigma"], int(cfg["n_particles"]), cfg["keep"])
    pf.label = (f"Particle filter (N={pf.n}, keep {pf.keep:.0%}, {cfg['setting'] / 100:.3f} m/s, "
                f"likelihood {cfg['like_sigma']:g} cm)")
    print(pf.label)
    test = data.test_set()
    st = evaluate(pf, test, out=OUT / "best", extra_params={"like_sigma_cm": cfg["like_sigma"]})
    print(st[st["step"].isin([1, 10, 50, 100, 200, 500])][["horizon_s", "mean_cm", "median_cm", "p95_cm", "max_cm"]]
          .round(2).to_string(index=False))
    convergence([pf, KalmanCV(data.measurement_noise(data.train_set()))], test)


if __name__ == "__main__":
    main()
