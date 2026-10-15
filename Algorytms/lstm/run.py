"""Evaluate the LSTM vs GRU study on the benchmark and write the results table and figures.

All models are scored on the same test windows: every measured frame of the 20 benchmark trajectories that has
at least 100 displacements of history (enough for the longest window, 100 frames), horizons 10 ms ... 5 s,
measured targets only. Errors in cm on positions (after undoing the normalisation).

Per model and seed (Results/lstm/<variant>/seed<k>/eval.json): mean error per horizon (pooled over all
origins), ADE 0.1 s (mean error over the 10 frames of the first block) and FDE 0.1 s (error at the 10th frame).
The table shows mean ± std over the seeds; the figures the mean over seeds with the min-max range.

Decision rule: a variant differs from the GRU baseline only when |difference of the means| is larger than the
spread between seeds (the larger of the two standard deviations); otherwise the GRU baseline is kept.

Usage (from the repo root; seeds not trained yet are skipped, so it can be run on partial results):
    .venv/Scripts/python -m Algorytms.lstm.run [--jobs 4]   # evaluate new runs (in parallel), table and figures
    .venv/Scripts/python -m Algorytms.lstm.run --timing     # also (re)measure the laptop call time; only when
                                                            # nothing else runs (training would distort it)
    .venv/Scripts/python -m Algorytms.lstm.run --best       # the winner (LSTM B, seed with the lowest validation
                                                            # loss) with the common protocol -> Results/lstm/best/
Writes Results/lstm/: results.md, error_vs_horizon.png, relative_to_gru.png, timing_laptop.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from Algorytms.baselines.models import ConstantVelocity
from Algorytms.common import data
from Algorytms.common.data import FRAME_S
from Algorytms.common.evaluation import evaluate, trajectory_errors
from Algorytms.common.plots import INK, INK2, MUTED, SERIES, SURFACE, _fig, _style, plot_training_curve, plt
from Algorytms.lstm.bench_cpu import bench
from Algorytms.lstm.model import N_IN, SEEDS, VARIANTS, RNNNet, load, n_params, weights_bytes
from Algorytms.lstm.train import OUT_DIR

MAX_H = 500
MIN_HISTORY = 101          # positions incl. the origin = 100 displacements, the longest window
CV = "constant_velocity"
CV_LABEL = "constant velocity"
BASE = "gru_base"
# winner of the study (beats the GRU baseline on every metric by more than the seed spread); enters the final benchmark
BEST_VARIANT = "lstm_b"
BEST_LABEL = "LSTM (velocity + acceleration input, hidden 32, 10-step output)"
METRICS = [("ade", "ADE 0.1 s"), ("fde", "FDE 0.1 s"), ("e050", "0.5 s"), ("e100", "1 s"), ("e500", "5 s")]


def score(model, test):
    err = np.concatenate([trajectory_errors(model, tr, MAX_H, MIN_HISTORY)[1] for tr in test])
    curve = np.nanmean(err, axis=0)
    return dict(n_origins=len(err), ade=float(np.nanmean(err[:, :10])), fde=float(curve[9]),
                e050=float(curve[49]), e100=float(curve[99]), e500=float(curve[499]), curve=curve.tolist())


def evaluate_one(job):
    """Score one trained run (variant, seed) and save its eval.json; runs in a worker process, one CPU thread."""
    v, s = job
    torch.set_num_threads(1)
    d = OUT_DIR / v / f"seed{s}"
    res = score(load(d / "model.pt", f"{v}_seed{s}", VARIANTS[v]["label"]), data.test_set())
    (d / "eval.json").write_text(json.dumps(res), encoding="utf-8")
    return f"{v} seed {s}: ADE {res['ade']:.3f}  FDE {res['fde']:.3f}  1 s {res['e100']:.2f} cm"


def evaluate_all(test, jobs=1):
    if not (OUT_DIR / CV / "eval.json").exists():
        (OUT_DIR / CV).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / CV / "eval.json").write_text(json.dumps(score(ConstantVelocity(), test)), encoding="utf-8")
    todo = [(v, s) for v in VARIANTS for s in SEEDS
            if (OUT_DIR / v / f"seed{s}" / "model.pt").exists() and not (OUT_DIR / v / f"seed{s}" / "checkpoint.pt").exists()
            and not (OUT_DIR / v / f"seed{s}" / "eval.json").exists()]
    with ProcessPoolExecutor(max(1, min(jobs, len(todo) or 1))) as pool:
        for line in pool.map(evaluate_one, todo):
            print(line, flush=True)


def collect():
    """{name: {"label", "runs": [eval dicts], "epochs": [...]}} for the models with at least one evaluated seed."""
    out = {CV: dict(label=CV_LABEL, runs=[json.loads((OUT_DIR / CV / "eval.json").read_text())], epochs=[])}
    for v, cfg in VARIANTS.items():
        runs, epochs = [], []
        for s in SEEDS:
            d = OUT_DIR / v / f"seed{s}"
            if (d / "eval.json").exists():
                runs.append(json.loads((d / "eval.json").read_text()))
                ck = torch.load(d / "model.pt", weights_only=False)
                h = [e["val_loss"] for e in ck["history"]]
                epochs.append((len(h), int(np.argmin(h)) + 1))
        if runs:
            out[v] = dict(label=cfg["label"], runs=runs, epochs=epochs)
    return out


def mean_std(runs, key):
    x = np.array([r[key] for r in runs])
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else np.nan)


def fmt(m, s, digits):
    return f"{m:.{digits}f}" + ("" if np.isnan(s) else f" ± {s:.{digits}f}")


def verdict(res, a, b, key):
    """Compare model a against b on one metric with the seed-spread rule."""
    ma, sa = mean_std(res[a]["runs"], key)
    mb, sb = mean_std(res[b]["runs"], key)
    spread = np.nanmax([sa, sb]) if not (np.isnan(sa) and np.isnan(sb)) else np.nan
    diff = ma - mb
    if np.isnan(spread):
        word = "n/a (one seed)"
    elif abs(diff) <= spread:
        word = "no difference"
    else:
        word = "**better**" if diff < 0 else "worse"
    return f"{diff:+.3f} (spread {spread:.3f}) {word}" if not np.isnan(spread) else f"{diff:+.3f} {word}"


def plot_curves(res, path, relative=False, title="LSTM vs GRU", ylim=(0.8, 1.6)):
    fig, ax = _fig(figsize=(11, 5.8))
    h = np.arange(1, MAX_H + 1) * FRAME_S
    ref = np.mean([r["curve"] for r in res[BASE]["runs"]], axis=0) if relative else 1.0
    colors = iter(SERIES)
    for name, m in res.items():
        c = np.array([r["curve"] for r in m["runs"]]) / ref
        color, ls = (MUTED, "--") if name == CV else (next(colors), "-")
        label = m["label"] + (f" ({len(c)} seed{'s' * (len(c) > 1)})" if name != CV and len(c) != len(SEEDS) else "")
        if len(c) > 1:
            ax.fill_between(h, c.min(0), c.max(0), color=color, alpha=0.18, lw=0)
        ax.plot(h, c.mean(0), color=color, lw=1.8, ls=ls, label=label)
    ax.set_xscale("log")
    ax.set_xlim(h[0], h[-1])
    ax.set_xlabel("prediction horizon [s] (log scale)", color=INK2)
    if relative:
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_ylabel("mean error / mean error of the GRU baseline", color=INK2)
        ax.set_title("Error relative to the GRU baseline (mean of seeds, band = min–max over seeds)", color=INK, loc="left")
    else:
        ax.set_yscale("log")
        ax.set_ylabel("mean position error [cm] (log scale)", color=INK2)
        ax.set_title(f"{title}: mean error vs horizon (mean of seeds, band = min–max over seeds)", color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def write_table(res, timing, path):
    rasp_path = OUT_DIR / "timing_raspberry.json"
    rasp = json.loads(rasp_path.read_text()) if rasp_path.exists() else None
    machines = "; ".join(f"{k}: {t['machine']}" for k, t in (("laptop", timing), ("Raspberry Pi", rasp)) if t)
    n = res[BASE]["runs"][0]["n_origins"]
    lines = ["# LSTM vs GRU: results\n",
             f"Benchmark: 20 test trajectories, {n} prediction origins (measured frames with ≥ 1 s = 100 displacements "
             "of history, the same for every model), measured targets only. Errors in cm. Every cell: mean ± std over "
             "the seeds (number of seeds in the `seeds` column). Time: one network call, batch 1, one CPU thread "
             f"({machines or 'not measured yet'}).\n",
             "| model | seeds | param. | ADE 0.1 s | FDE 0.1 s | 0.5 s | 1 s | 5 s | time laptop [ms] | "
             "time Raspberry [ms] | weights [kB] | epochs (best) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in res.items():
        cells = [fmt(*mean_std(m["runs"], k), 3 if k in ("ade", "fde") else 2) for k, _ in METRICS]
        if name == CV:
            lines.append(f"| {m['label']} | – | 0 | " + " | ".join(cells) + " | – | – | – | – |")
            continue
        v = VARIANTS[name]
        net = RNNNet(v["cell"], N_IN[v["inp"]], v["hidden"])
        lap = f"{timing['variants'][name]['call_ms']:.3f}" if timing else "not measured yet"
        r = f"{rasp['variants'][name]['call_ms']:.3f}" if rasp else "not measured yet"
        ep = np.array(m["epochs"])
        lines.append(f"| {m['label']} | {len(m['runs'])} | {n_params(net)} | " + " | ".join(cells)
                     + f" | {lap} | {r} | {weights_bytes(net) / 1024:.1f} | "
                       f"{ep[:, 0].mean():.0f} ({ep[:, 1].mean():.0f}) |")
    lines += ["\n## Decision rule: each variant against the GRU baseline\n",
              "Difference of the means (variant − GRU baseline) in cm; the variant counts as better or worse only "
              "when the difference is larger than the spread between seeds (larger of the two std). Negative = "
              "smaller error than the GRU baseline.\n",
              "| variant | " + " | ".join(lbl for _, lbl in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name in res:
        if name not in (CV, BASE):
            lines.append(f"| {res[name]['label']} | " + " | ".join(verdict(res, name, BASE, k) for k, _ in METRICS) + " |")
    pairs = [("lstm_a28", "lstm_a", "is the gain of LSTM A due to LSTM or to the larger network? (A' vs A)"),
             ("lstm_b", "lstm_a", "does the acceleration input help the LSTM? (B vs A)"),
             ("gru_b", BASE, "does the acceleration input help the GRU? (GRU-B vs GRU baseline)"),
             ("lstm_b", "gru_b", "same 6 inputs: LSTM vs GRU (B vs GRU-B)"),
             ("lstm_c50", "lstm_a", "longer window, 50 vs 20 (C50 vs A)"),
             ("lstm_c100", "lstm_a", "longer window, 100 vs 20 (C100 vs A)")]
    pairs = [p for p in pairs if p[0] in res and p[1] in res]
    if pairs:
        lines += ["\n## Specific questions (first model − second model)\n",
                  "| question | " + " | ".join(lbl for _, lbl in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        lines += [f"| {q} | " + " | ".join(verdict(res, a, b, k) for k, _ in METRICS) + " |" for a, b, q in pairs]
    lines += ["\n![error vs horizon](error_vs_horizon.png)\n", "![relative to GRU](relative_to_gru.png)\n"]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_best(test, variant=BEST_VARIANT):
    """The winner of the study as one model for the final benchmark: the seed with the lowest validation loss
    (never chosen on the test set), evaluated with the common protocol of every algorithm -> Results/lstm/best/."""
    val = {s: min(e["val_loss"] for e in json.loads((OUT_DIR / variant / f"seed{s}" / "history.json").read_text())["history"])
           for s in SEEDS}
    seed = min(val, key=val.get)
    out = OUT_DIR / "best"
    out.mkdir(exist_ok=True)
    ck = torch.load(OUT_DIR / variant / f"seed{seed}" / "model.pt", weights_only=False)
    model = load(OUT_DIR / variant / f"seed{seed}" / "model.pt", "lstm_best", BEST_LABEL)
    best_epoch = int(np.argmin([e["val_loss"] for e in ck["history"]])) + 1
    plot_training_curve(ck["history"], best_epoch, f"{BEST_LABEL}, seed {seed}: training", out / "training_curve.png")
    st = evaluate(model, test, extra_params=dict(variant=variant, seed=seed, selected_by="lowest validation loss",
                                                 val_loss_per_seed=val, best_epoch=best_epoch), out=out)
    print(f"{variant}: seed {seed} (validation loss per seed: " + ", ".join(f"{s}: {v:.5f}" for s, v in val.items()) + ")")
    print(st[st["step"].isin([1, 10, 50, 100, 500])][["horizon_s", "mean_cm", "median_cm", "max_cm"]]
          .round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="models evaluated in parallel (one CPU thread each)")
    ap.add_argument("--timing", action="store_true", help="measure the call time on this machine as the laptop time")
    ap.add_argument("--best", action="store_true", help="only: evaluate the selected model for the final benchmark")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    test = data.test_set()
    if args.best:
        torch.set_num_threads(4)
        return evaluate_best(test)
    evaluate_all(test, args.jobs)
    timing_path = OUT_DIR / "timing_laptop.json"
    if args.timing:
        timing_path.write_text(json.dumps(bench(), indent=1), encoding="utf-8")
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    res = collect()
    if BASE not in res:
        print("no evaluated GRU baseline yet")
        return
    plot_curves(res, OUT_DIR / "error_vs_horizon.png")
    plot_curves(res, OUT_DIR / "relative_to_gru.png", relative=True)
    write_table(res, timing, OUT_DIR / "results.md")
    print((OUT_DIR / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
