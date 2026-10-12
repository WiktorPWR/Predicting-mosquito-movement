"""Evaluate trained GRU models on the benchmark (same protocol as every algorithm).

Usage (from the repo root, after training):
    .venv/Scripts/python -m Algorytms.gru.run              # every Results/gru/*/model.pt
    .venv/Scripts/python -m Algorytms.gru.run disp_h32     # one model
"""
import sys

import torch

from Algorytms.common import data
from Algorytms.common.data import RESULTS_DIR
from Algorytms.common.evaluation import evaluate
from Algorytms.gru.model import load


def main():
    torch.set_num_threads(4)
    dirs = [RESULTS_DIR / "gru" / n for n in sys.argv[1:]] or sorted((RESULTS_DIR / "gru").glob("*/model.pt"))
    test = data.test_set()
    for d in dirs:
        d = d.parent if d.name == "model.pt" else d
        model = load(d / "model.pt", name=f"gru_{d.name}")     # label from the checkpoint
        st = evaluate(model, test, out=d)
        print(model.label)
        print(st[st["step"].isin([1, 10, 50, 100, 500])][["horizon_s", "mean_cm", "median_cm", "max_cm"]]
              .round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
