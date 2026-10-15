"""Evaluate the transformer variants and compare them with the GRU baseline and the best LSTM (LSTM B).

Same test windows and metrics as the LSTM vs GRU study (Algorytms/lstm/run.py): every measured frame of the 20
benchmark trajectories with at least 100 displacements of history, horizons 10 ms ... 5 s, measured targets only;
ADE / FDE of the first 0.1 s block and the rollout error at 0.5 / 1 / 5 s. The GRU baseline, LSTM B and constant
velocity are read from Results/lstm/ (already evaluated with 5 seeds on exactly these windows), not retrained.

Decision rule: a transformer variant differs from a reference only when |difference of the means| is larger than
the spread between seeds (the larger of the two standard deviations).

Usage (from the repo root; runs not trained yet are skipped, so it works on partial results):
    python -m Algorytms.transformer.run [--jobs 4]
    python -m Algorytms.transformer.run --best   # the winner (T3, seed with the lowest validation loss) with the
                                                 # common protocol of every algorithm -> Results/transformer/best/
Writes Results/transformer/: results.md, error_vs_horizon.png, relative_to_gru.png;
per run Results/transformer/<variant>/seed<k>/eval.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from Algorytms.common import data
from Algorytms.common.evaluation import evaluate
from Algorytms.common.plots import plot_training_curve
from Algorytms.lstm import run as lstm_run
from Algorytms.lstm.model import N_IN, VARIANTS as LSTM_VARIANTS, RNNNet, n_params, weights_bytes
from Algorytms.lstm.run import BASE, CV, CV_LABEL, METRICS, fmt, mean_std, plot_curves, score, verdict
from Algorytms.lstm.train import OUT_DIR as LSTM_DIR
from Algorytms.transformer.model import SEEDS, VARIANTS, KomarTF, arch, load
from Algorytms.transformer.train import OUT_DIR

REFS = (BASE, "lstm_b")       # reference models from the LSTM study
# winner of the study: better than T1 on 4 of 5 metrics (equal at 5 s) by more than the seed spread; T2 never
# better than T1 despite 2.3x the parameters
BEST_VARIANT = "tf_t3"
BEST_LABEL = "Transformer (velocity input, window 1 s in 10 patches, d 20, 10-step output)"


def evaluate_one(job):
    v, s = job
    torch.set_num_threads(1)
    d = OUT_DIR / v / f"seed{s}"
    res = score(load(d / "model.pt", f"{v}_seed{s}", VARIANTS[v]["label"]), data.test_set())
    (d / "eval.json").write_text(json.dumps(res), encoding="utf-8")
    return f"{v} seed {s}: ADE {res['ade']:.3f}  FDE {res['fde']:.3f}  1 s {res['e100']:.2f} cm"


def evaluate_all(jobs):
    todo = [(v, s) for v in VARIANTS for s in SEEDS
            if (OUT_DIR / v / f"seed{s}" / "model.pt").exists() and not (OUT_DIR / v / f"seed{s}" / "checkpoint.pt").exists()
            and not (OUT_DIR / v / f"seed{s}" / "eval.json").exists()]
    with ProcessPoolExecutor(max(1, min(jobs, len(todo) or 1))) as pool:
        for line in pool.map(evaluate_one, todo):
            print(line, flush=True)


def runs_of(folder):
    runs, epochs = [], []
    for s in SEEDS:
        d = folder / f"seed{s}"
        if (d / "eval.json").exists():
            runs.append(json.loads((d / "eval.json").read_text()))
            h = [e["val_loss"] for e in json.loads((d / "history.json").read_text())["history"]]
            epochs.append((len(h), int(np.argmin(h)) + 1))
    return runs, epochs


def collect():
    res = {}
    if (LSTM_DIR / CV / "eval.json").exists():
        res[CV] = dict(label=CV_LABEL, runs=[json.loads((LSTM_DIR / CV / "eval.json").read_text())], epochs=[])
    for name in REFS:
        runs, epochs = runs_of(LSTM_DIR / name)
        if runs:
            res[name] = dict(label=LSTM_VARIANTS[name]["label"], runs=runs, epochs=epochs, ref=True)
    for name, v in VARIANTS.items():
        runs, epochs = runs_of(OUT_DIR / name)
        if runs:
            res[name] = dict(label=v["label"], runs=runs, epochs=epochs)
    return res


def timing(folder, kind):
    f = folder / f"timing_{kind}.json"
    return json.loads(f.read_text())["variants"] if f.exists() else {}


def write_table(res, path):
    lap = {**timing(LSTM_DIR, "laptop"), **timing(OUT_DIR, "laptop")}
    rasp = {**timing(LSTM_DIR, "raspberry"), **timing(OUT_DIR, "raspberry")}
    n = next(iter(res.values()))["runs"][0]["n_origins"]
    lines = ["# Transformer vs GRU / LSTM: results\n",
             f"Benchmark: 20 test trajectories, {n} prediction origins (measured frames with ≥ 1 s = 100 displacements "
             "of history, the same for every model), measured targets only. Errors in cm. Every cell: mean ± std over "
             "the seeds. GRU baseline and LSTM B come from the LSTM vs GRU study (Results/lstm/), same windows. "
             "Time: one network call, batch 1, one CPU thread.\n",
             "| model | seeds | param. | ADE 0.1 s | FDE 0.1 s | 0.5 s | 1 s | 5 s | time laptop [ms] | "
             "time Raspberry [ms] | weights [kB] | epochs (best) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in res.items():
        cells = [fmt(*mean_std(m["runs"], k), 3 if k in ("ade", "fde") else 2) for k, _ in METRICS]
        if name == CV:
            lines.append(f"| {m['label']} | – | 0 | " + " | ".join(cells) + " | – | – | – | – |")
            continue
        if name in REFS:
            v = LSTM_VARIANTS[name]
            net = RNNNet(v["cell"], N_IN[v["inp"]], v["hidden"])
        else:
            net = KomarTF(**arch(name))
        t = lambda d: f"{d[name]['call_ms']:.3f}" if name in d else "not measured yet"
        ep = np.array(m["epochs"])
        lines.append(f"| {m['label']} | {len(m['runs'])} | {n_params(net)} | " + " | ".join(cells)
                     + f" | {t(lap)} | {t(rasp)} | {weights_bytes(net) / 1024:.1f} | "
                       f"{ep[:, 0].mean():.0f} ({ep[:, 1].mean():.0f}) |")
    tf = [n for n in VARIANTS if n in res]
    for ref in [r for r in REFS if r in res]:
        lines += [f"\n## Decision rule: each transformer against {res[ref]['label']}\n",
                  f"Difference of the means (transformer − {res[ref]['label']}) in cm; better / worse only when larger "
                  "than the spread between seeds (larger of the two std). Negative = smaller error.\n",
                  "| variant | " + " | ".join(lbl for _, lbl in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        lines += [f"| {res[n]['label']} | " + " | ".join(verdict(res, n, ref, k) for k, _ in METRICS) + " |" for n in tf]
    pairs = [("tf_t2", "tf_t1", "larger transformer, d 32 vs d 20 (T2 vs T1)"),
             ("tf_t3", "tf_t1", "1 s window with patches vs 0.2 s window (T3 vs T1)")]
    pairs = [p for p in pairs if p[0] in res and p[1] in res]
    if pairs:
        lines += ["\n## Specific questions (first model − second model)\n",
                  "| question | " + " | ".join(lbl for _, lbl in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        lines += [f"| {q} | " + " | ".join(verdict(res, a, b, k) for k, _ in METRICS) + " |" for a, b, q in pairs]
    lines += ["\n![error vs horizon](error_vs_horizon.png)\n", "![relative to GRU](relative_to_gru.png)\n"]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_best(variant=BEST_VARIANT):
    """The winner as one model for the final benchmark: the seed with the lowest validation loss (never chosen on
    the test set), evaluated with the common protocol of every algorithm -> Results/transformer/best/."""
    val = {s: min(e["val_loss"] for e in json.loads((OUT_DIR / variant / f"seed{s}" / "history.json").read_text())["history"])
           for s in SEEDS}
    seed = min(val, key=val.get)
    out = OUT_DIR / "best"
    out.mkdir(exist_ok=True)
    ck = torch.load(OUT_DIR / variant / f"seed{seed}" / "model.pt", weights_only=False)
    best_epoch = int(np.argmin([e["val_loss"] for e in ck["history"]])) + 1
    plot_training_curve(ck["history"], best_epoch, f"{BEST_LABEL}, seed {seed}: training", out / "training_curve.png")
    model = load(OUT_DIR / variant / f"seed{seed}" / "model.pt", "transformer_best", BEST_LABEL)
    st = evaluate(model, data.test_set(), extra_params=dict(variant=variant, seed=seed, selected_by="lowest validation loss",
                                                            val_loss_per_seed=val, best_epoch=best_epoch), out=out)
    print(f"{variant}: seed {seed} (validation loss per seed: " + ", ".join(f"{s}: {v:.5f}" for s, v in val.items()) + ")")
    print(st[st["step"].isin([1, 10, 50, 100, 500])][["horizon_s", "mean_cm", "median_cm", "max_cm"]]
          .round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="models evaluated in parallel (one CPU thread each)")
    ap.add_argument("--best", action="store_true", help="only: evaluate the selected model for the final benchmark")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.best:
        torch.set_num_threads(4)
        return evaluate_best()
    evaluate_all(args.jobs)
    res = collect()
    if BASE not in res or not any(n in res for n in VARIANTS):
        print("nothing to compare yet (needs Results/lstm/gru_base/*/eval.json and at least one evaluated transformer)")
        return
    lstm_run.SEEDS = SEEDS
    plot_curves(res, OUT_DIR / "error_vs_horizon.png", title="Transformer vs GRU / LSTM")
    plot_curves(res, OUT_DIR / "relative_to_gru.png", relative=True, ylim=(0.5, 1.4))
    write_table(res, OUT_DIR / "results.md")
    print((OUT_DIR / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
