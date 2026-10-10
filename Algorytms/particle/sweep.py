"""Parameter sweeps for the particle filter (run in parallel processes).

Stage "noise": process-noise setting x likelihood width, N = NOISE_N particles, keep = 100 %.
    setting:     v_max * k / 10, k = 1..10 (as for the Kalman filter)
    like_sigma:  measurement std assumed by the likelihood [cm]. The real noise
                 (~0.01-0.05 cm) makes the weights so peaked that one particle
                 takes all the weight every frame; a wider likelihood keeps
                 more particles alive and averages their velocities over time.
Stage "grid": number of particles x elite fraction with the best noise pair.
    n_particles: 100, 250, 500, 1000, 2500, 5000
    keep:        1, 5, 10, 25, 50, 100 %

Best = lowest mean ratio of mean error to the best configuration over TABLE_STEPS.
Results/particle/sweep/: <stage>_by_horizon.csv (mean error for all 500 horizons
per configuration, + runtime), <stage>.md tables, heatmaps.

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.particle.sweep noise
    .venv/Scripts/python -m Algorytms.particle.sweep grid
    .venv/Scripts/python -m Algorytms.particle.sweep plot      # redraw from CSVs
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from Algorytms.common import data
from Algorytms.common.data import FRAME_S, RESULTS_DIR
from Algorytms.common.evaluation import TABLE_STEPS, score
from Algorytms.common.plots import plot_heatmap, plot_particles_effect
from Algorytms.particle.particle_filter import INTERP_STD_CM, ParticleFilter

SWEEP_DIR = RESULTS_DIR / "particle" / "sweep"
LIKE_SIGMAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
NOISE_N = 1000
N_PARTICLES = [100, 250, 500, 1000, 2500, 5000]
KEEPS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
WORKERS = 3
H_COLS = [f"h{s}" for s in range(1, 501)]

_test = None


def _init():
    global _test
    _test = data.test_set()


def make_filter(setting, like_sigma, n_particles, keep, seed=0):
    return ParticleFilter(np.full(3, like_sigma), setting, n_particles, keep, seed=seed,
                          interp_std=max(INTERP_STD_CM, like_sigma))


def _run(cfg):
    t = time.time()
    st = score(make_filter(**cfg), _test)[0]
    return {**cfg, "runtime_s": time.time() - t, **dict(zip(H_COLS, st["mean_cm"].to_numpy()))}


def run_configs(cfgs, csv):
    rows = []
    with ProcessPoolExecutor(WORKERS, initializer=_init) as ex:
        for r in ex.map(_run, cfgs):
            rows.append(r)
            print(f"  {', '.join(f'{k}={r[k]:g}' for k in cfgs[0])}: {r['runtime_s']:.0f} s, "
                  f"mean error 0.1 s {r['h10']:.2f} cm, 1 s {r['h100']:.2f} cm", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return df


def rank(df):
    """Adds the selection criterion: mean over TABLE_STEPS of error / best error at that horizon."""
    cols = [f"h{s}" for s in TABLE_STEPS]
    df = df.copy()
    df["rel_to_best"] = (df[cols] / df[cols].min()).mean(axis=1)
    return df


def write_table(df, keys, path, title):
    df = rank(df).sort_values(keys)
    cols = [f"h{s}" for s in TABLE_STEPS]
    best = df[cols].min()
    names = {"setting": "setting [m/s]", "like_sigma": "likelihood sigma [cm]", "n_particles": "particles",
             "keep": "kept best [%]"}
    fmt = {"setting": lambda v: f"{v / 100:.3f}", "keep": lambda v: f"{v * 100:g}"}
    show = lambda k, v: fmt.get(k, lambda x: f"{x:g}")(v)
    head = [names[k] for k in keys] + [f"{s * FRAME_S:g}s" for s in TABLE_STEPS] + ["runtime [s]", "error / best"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        cells = [show(k, r[k]) for k in keys]
        cells += [f"**{r[c]:.2f}**" if r[c] == best[c] else f"{r[c]:.2f}" for c in cols]
        cells += [f"{r['runtime_s']:.0f}", f"{r['rel_to_best']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    winner = df.loc[df["rel_to_best"].idxmin()]
    path.write_text(f"# {title}\n\nMean position error [cm] per prediction horizon on the test set; bold = best "
                    f"for that horizon. Runtime = whole test set, one process. Last column: mean ratio to the best "
                    f"configuration over these horizons (lower is better).\n\n**Best: "
                    + ", ".join(f"{names[k]} = {show(k, winner[k])}" for k in keys) + "**\n\n" + "\n".join(lines) + "\n",
                    encoding="utf-8")
    return winner


def best_noise():
    df = rank(pd.read_csv(SWEEP_DIR / "noise_by_horizon.csv"))
    w = df.loc[df["rel_to_best"].idxmin()]
    return float(w["setting"]), float(w["like_sigma"])


def plots():
    f = SWEEP_DIR / "noise_by_horizon.csv"
    if f.exists():
        df = pd.read_csv(f).copy()
        write_table(df, ["setting", "like_sigma"], SWEEP_DIR / "noise.md",
                    f"Particle filter: process noise x likelihood width (N = {NOISE_N}, keep 100 %)")
        for s in (10, 100):
            plot_heatmap(df.assign(setting=(df["setting"] / 100).round(3)), "like_sigma", "setting", f"h{s}",
                         "likelihood sigma [cm]", "process-noise setting [m/s]",
                         f"Particle filter: mean error at {s * FRAME_S:g} s horizon [cm]",
                         SWEEP_DIR / f"noise_heatmap_{s * FRAME_S:g}s.png")
    f = SWEEP_DIR / "grid_by_horizon.csv"
    if f.exists():
        df = pd.read_csv(f).copy()
        s0, l0 = df["setting"].iat[0], df["like_sigma"].iat[0]
        write_table(df, ["n_particles", "keep"], SWEEP_DIR / "grid.md",
                    f"Particle filter: number of particles x elite fraction "
                    f"(setting {s0 / 100:.3f} m/s, likelihood sigma {l0:g} cm)")
        d = df.assign(keep_pct=(df["keep"] * 100).round(0).astype(int))
        for s in (1, 10, 100):
            plot_heatmap(d, "n_particles", "keep_pct", f"h{s}", "number of particles", "kept best particles [%]",
                         f"Particle filter: mean error at {s * FRAME_S:g} s horizon [cm]",
                         SWEEP_DIR / f"grid_heatmap_{s * FRAME_S:g}s.png")
        plot_heatmap(d, "n_particles", "keep_pct", "runtime_s", "number of particles", "kept best particles [%]",
                     "Particle filter: runtime on the whole test set [s]", SWEEP_DIR / "grid_runtime.png", fmt="{:.0f}",
                     clip=None)
        plot_particles_effect(df, SWEEP_DIR / "grid_particles_effect.png")


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if stage == "noise":
        v_max = data.max_speed(data.train_set())
        cfgs = [dict(setting=v_max * k / 10, like_sigma=l, n_particles=NOISE_N, keep=1.0)
                for k in range(1, 11) for l in LIKE_SIGMAS]
        run_configs(cfgs, SWEEP_DIR / "noise_by_horizon.csv")
    elif stage == "grid":
        setting, like_sigma = best_noise()
        print(f"grid with setting {setting / 100:.3f} m/s, likelihood sigma {like_sigma:g} cm")
        cfgs = [dict(setting=setting, like_sigma=like_sigma, n_particles=n, keep=k)
                for n in sorted(N_PARTICLES, reverse=True) for k in KEEPS]   # slowest first for load balance
        run_configs(cfgs, SWEEP_DIR / "grid_by_horizon.csv")
    plots()
    for name in ("noise", "grid"):
        f = SWEEP_DIR / f"{name}_by_horizon.csv"
        if f.exists():
            df = rank(pd.read_csv(f))
            w = df.loc[df["rel_to_best"].idxmin()]
            print(f"best {name}:", {k: w[k] for k in ("setting", "like_sigma", "n_particles", "keep")})
    if (SWEEP_DIR / "grid_by_horizon.csv").exists():
        df = rank(pd.read_csv(SWEEP_DIR / "grid_by_horizon.csv"))
        w = df.loc[df["rel_to_best"].idxmin()]
        (SWEEP_DIR / "best.json").write_text(json.dumps(
            {k: float(w[k]) for k in ("setting", "like_sigma", "n_particles", "keep")}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
