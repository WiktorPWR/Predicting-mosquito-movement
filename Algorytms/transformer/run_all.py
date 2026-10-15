"""Train every transformer variant x seed, N_PARALLEL runs at a time (1 CPU thread each), slowest first.

Finished runs are skipped and interrupted runs continue from their checkpoint, so the command can be stopped at
any time and started again. Logs: Results/transformer/logs/<variant>_seed<k>.log (not in git).

Usage (from the repo root):
    python -m Algorytms.transformer.run_all [--parallel 6]
    python -m Algorytms.transformer.run_all status          # progress of every run
"""
import argparse
import json
import subprocess
import sys
import time

from Algorytms.transformer.model import SEEDS, VARIANTS
from Algorytms.transformer.train import OUT_DIR

COST = {"tf_t2": 3, "tf_t1": 2, "tf_t3": 1}      # relative epoch time measured on the laptop


def jobs():
    return sorted(((v, s) for s in SEEDS for v in VARIANTS), key=lambda j: -COST[j[0]])


def state(v, s):
    d = OUT_DIR / v / f"seed{s}"
    if (d / "model.pt").exists() and not (d / "checkpoint.pt").exists():
        return "done", json.loads((d / "history.json").read_text())["history"] if (d / "history.json").exists() else []
    if (d / "progress.json").exists():
        return "running", json.loads((d / "progress.json").read_text())
    return "waiting", None


def status():
    n_done = 0
    for v, s in jobs():
        st, info = state(v, s)
        n_done += st == "done"
        extra = (f"{len(info)} epochs" if st == "done" else
                 f"epoch {info['epochs']}/{info['max_epochs']} (no improvement for {info['bad']})" if st == "running" else "")
        print(f"{v:6s} seed {s}  {st:8s} {extra}")
    print(f"{n_done}/{len(jobs())} runs finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="train", choices=["train", "status"])
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    if args.what == "status":
        return status()
    logs = OUT_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if state(*j)[0] != "done"]
    running = []
    while todo or running:
        running = [(j, p) for j, p in running if p.poll() is None]
        while todo and len(running) < args.parallel:
            v, s = todo.pop(0)
            log = open(logs / f"{v}_seed{s}.log", "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "Algorytms.transformer.train", v, "--seed", str(s)],
                                 stdout=log, stderr=subprocess.STDOUT)
            running.append(((v, s), p))
            print(f"{time.strftime('%H:%M')} started {v} seed {s}", flush=True)
        time.sleep(20)
    print(f"{time.strftime('%H:%M')} all runs finished", flush=True)


if __name__ == "__main__":
    main()
