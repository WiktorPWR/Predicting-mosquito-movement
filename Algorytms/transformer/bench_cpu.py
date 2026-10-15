"""CPU time of one network call (batch 1, one 0.1 s block) for every transformer variant.

Freshly initialised networks (weights do not change the time); runs anywhere with numpy and torch, e.g. on the
Raspberry Pi. Same method as Algorytms/lstm/bench_cpu.py (median of 2 000 calls, one thread).

Usage (from the repo root):
    python -m Algorytms.transformer.bench_cpu
    python -m Algorytms.transformer.bench_cpu --save Results/transformer/timing_laptop.json      # laptop
    python -m Algorytms.transformer.bench_cpu --save Results/transformer/timing_raspberry.json   # Raspberry Pi
"""
import argparse
import json
import platform

import torch

from Algorytms.lstm.bench_cpu import call_ms
from Algorytms.lstm.model import n_params, weights_bytes
from Algorytms.transformer.model import VARIANTS, KomarTF, arch


def bench():
    rows = {}
    for name, v in VARIANTS.items():
        torch.manual_seed(0)
        net = KomarTF(**arch(name))
        rows[name] = dict(params=n_params(net), weights_bytes=weights_bytes(net), call_ms=call_ms(net, v["window"], 3))
    return {"machine": f"{platform.node()} {platform.machine()} {platform.processor()}".strip(),
            "torch": torch.__version__, "variants": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    res = bench()
    print(res["machine"], "torch", res["torch"])
    for name, r in res["variants"].items():
        print(f"{name:6s} {r['params']:6d} params  {r['weights_bytes'] / 1024:5.1f} kB  {r['call_ms']:.3f} ms")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
