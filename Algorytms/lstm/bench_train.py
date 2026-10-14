"""CPU speed of one training step (forward + backward + Adam step), to compare machines and thread settings.

Configurations: the 7 study variants as they are trained (batch 64) and a grid of larger LSTMs (window 20,
hidden 64/128/256 x batch 32/64/128). Every configuration runs with each thread count (default: 1 and all
cores): a few warm-up steps, then the mean time of the measured steps. Random input of the right shape, so no
data or trained model is needed.

Columns: ms per step; training part of one epoch of this study (940 143 windows / batch) in minutes;
estimated memory = weights + gradients + 2 Adam moments + activations kept for backward (rough).
"1 thread x N runs" vs "N threads x 1 run": with tiny networks parallel 1-thread runs use the CPU better;
the table shows by how much.

Usage (from the repo root):
    python -m Algorytms.lstm.bench_train [--threads 1,6] [--steps 100] [--save]
--save writes Results/lstm/bench_train_<computer name>.md
"""
import argparse
import os
import platform
import time

import torch

from Algorytms.common.data import RESULTS_DIR
from Algorytms.gru.train import BATCH, LR, masked_mse
from Algorytms.lstm.model import N_IN, STEPS, VARIANTS, RNNNet, n_params

N_TRAIN = 940143       # training windows of the study (window 20)


def configs():
    for name, v in VARIANTS.items():
        yield name, v["cell"], N_IN[v["inp"]], v["hidden"], v["window"], BATCH
    for hidden in (64, 128, 256):
        for batch in (32, 64, 128):
            yield "lstm", "lstm", 3, hidden, 20, batch


def memory_mb(net, cell, n_in, hidden, window, batch):
    gates = 4 if cell == "lstm" else 3
    weights = n_params(net) * 4 * 4                                # weights, grads, Adam m and v (float32)
    acts = batch * window * (n_in + (gates + 2) * hidden) * 4 * 2  # inputs, gates, states (+ backward buffers)
    return (weights + acts) / 2 ** 20


def step_ms(cell, n_in, hidden, window, batch, threads, warmup, steps):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    net = RNNNet(cell, n_in, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    X, Y = torch.randn(batch, window, n_in), torch.randn(batch, STEPS, 3)
    M = torch.ones(batch, STEPS, dtype=torch.bool)
    for i in range(warmup + steps):
        if i == warmup:
            t0 = time.perf_counter()
        loss = masked_mse(net(X), Y, M)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (time.perf_counter() - t0) / steps * 1000, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default=f"1,{os.cpu_count()}")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    threads = sorted({int(t) for t in args.threads.split(",")})
    head = (f"Training step benchmark: {platform.node()}, {platform.processor() or platform.machine()}, "
            f"{os.cpu_count()} cores, torch {torch.__version__}, device cpu\n")
    lines = [head, "| config | cell | input | hidden | window | batch | param. | threads | ms / step | "
                   "epoch [min] | memory est. [MB] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    print("\n".join(lines))
    for name, cell, n_in, hidden, window, batch in configs():
        for th in threads:
            ms, net = step_ms(cell, n_in, hidden, window, batch, th, args.warmup, args.steps)
            row = (f"| {name} | {cell} | {n_in} | {hidden} | {window} | {batch} | {n_params(net)} | {th} | "
                   f"{ms:.2f} | {ms * -(-N_TRAIN // batch) / 60000:.1f} | "
                   f"{memory_mb(net, cell, n_in, hidden, window, batch):.1f} |")
            lines.append(row)
            print(row, flush=True)
    if args.save:
        path = RESULTS_DIR / "lstm" / f"bench_train_{platform.node()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
