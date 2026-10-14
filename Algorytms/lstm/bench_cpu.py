"""CPU time of one network call (batch 1, one 0.1 s block = 10 displacements) for every variant.

Weights do not change the time, so freshly initialised networks are used: this runs anywhere with numpy and
torch, e.g. on the Raspberry Pi (copy the repository, no data or trained models needed).

Usage (from the repo root):
    python -m Algorytms.lstm.bench_cpu                              # print
    python -m Algorytms.lstm.bench_cpu --save Results/lstm/timing_raspberry.json   # on the Raspberry Pi
"""
import argparse
import json
import platform
import time

import numpy as np
import torch

from Algorytms.lstm.model import N_IN, VARIANTS, RNNNet, n_params, weights_bytes


@torch.no_grad()
def call_ms(net, window, n_in, warmup=200, n=2000):
    """Median wall time [ms] of net(x) for x of shape (1, window, n_in), one CPU thread."""
    torch.set_num_threads(1)
    net.eval()
    x = torch.randn(1, window, n_in)
    for _ in range(warmup):
        net(x)
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        net(x)
        t[i] = time.perf_counter() - t0
    return float(np.median(t) * 1000)


def bench():
    rows = {}
    for name, v in VARIANTS.items():
        torch.manual_seed(0)
        net = RNNNet(v["cell"], N_IN[v["inp"]], v["hidden"])
        rows[name] = dict(params=n_params(net), weights_bytes=weights_bytes(net),
                          call_ms=call_ms(net, v["window"], N_IN[v["inp"]]))
    return {"machine": f"{platform.node()} {platform.machine()} {platform.processor()}".strip(),
            "torch": torch.__version__, "variants": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    res = bench()
    print(res["machine"], "torch", res["torch"])
    for name, r in res["variants"].items():
        print(f"{name:10s} {r['params']:5d} params  {r['weights_bytes'] / 1024:5.1f} kB  {r['call_ms']:.3f} ms")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
