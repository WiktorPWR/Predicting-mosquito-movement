"""Train one transformer variant with one seed (CPU only).

Training is identical to the GRU / LSTM study (Algorytms/lstm/train.py): same split, windows and normalisation,
Adam lr 1e-3, batch 64, MSE over the measured target frames, at least 20 epochs, then stop after 5 epochs
without improvement of the validation loss (max 40), best epoch's weights kept. Dropout (0.1) is active in
training only; validation and the benchmark run the network in eval mode.

If the training turns out unstable (loss jumping), --lr and --warmup (linear learning-rate warm-up over the first
N optimiser steps) can be changed; any change goes to its own folder, outside the study table.

Usage (from the repo root):
    python -m Algorytms.transformer.train <variant> --seed <k> [--threads 1]
        variant: tf_t1 | tf_t2 | tf_t3   (see model.VARIANTS)
    Optional: --batch B  --max-epochs E  --lr LR  --warmup STEPS  --checkpoint PATH
Writes Results/transformer/<variant>[_b<B>][_e<E>][_lr<LR>][_wu<STEPS>]/seed<k>/: model.pt, history.json,
train.log (progress.json while training). A per-epoch checkpoint (model + optimiser + scheduler + epoch + random
state) lets an interrupted run continue exactly where it stopped: run the same command again.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from Algorytms.common.data import RESULTS_DIR
from Algorytms.common.ml_data import disp_std, load_split
from Algorytms.gru.train import BATCH, LR, STOPPING, masked_mse
from Algorytms.lstm.model import Normaliser
from Algorytms.lstm.train import Windows, evaluate_split
from Algorytms.transformer.model import VARIANTS, KomarTF, arch, save

OUT_DIR = RESULTS_DIR / "transformer"


def run_dir(variant, seed, batch, max_epochs, lr, warmup):
    suffix = "".join(f"_{k}{x:g}" for k, x, d in (("b", batch, BATCH), ("e", max_epochs, STOPPING["default"]["max_epochs"]),
                                                  ("lr", lr, LR), ("wu", warmup, 0)) if x != d)
    return OUT_DIR / (variant + suffix) / f"seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("variant", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 when several runs go in parallel)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-epochs", type=int, default=STOPPING["default"]["max_epochs"])
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warm-up over this many optimiser steps")
    ap.add_argument("--checkpoint", help="checkpoint file (default: checkpoint.pt in the run folder)")
    args = ap.parse_args()
    v, seed = VARIANTS[args.variant], args.seed
    stop = dict(STOPPING["default"], max_epochs=args.max_epochs,
                min_epochs=min(STOPPING["default"]["min_epochs"], args.max_epochs))
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = run_dir(args.variant, seed, args.batch, args.max_epochs, args.lr, args.warmup)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt" if args.checkpoint is None else Path(args.checkpoint)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    if (out / "model.pt").exists() and not ckpt_path.exists():
        log(f"{out.parent.name} seed {seed}: already trained, nothing to do")
        return

    train, val = load_split("train"), load_split("val")
    norm = Normaliser(disp_std())
    Wtr, Wva = Windows(train, norm, "disp", v["window"]), Windows(val, norm, "disp", v["window"])
    net = KomarTF(**arch(args.variant))
    log(f"{out.parent.name} seed {seed}: transformer {arch(args.variant)} batch {args.batch} lr {args.lr:g} "
        f"warm-up {args.warmup}, {sum(p.numel() for p in net.parameters())} parameters, {args.threads} thread(s); "
        f"train {len(Wtr)} windows, val {len(Wva)}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / args.warmup)) if args.warmup else None
    history, best, best_state, best_smooth, bad, start = [], np.inf, None, np.inf, 0, 1
    if ckpt_path.exists():                # resume an interrupted run exactly where it stopped
        ck = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
        history, best, best_state, best_smooth, bad = ck["history"], ck["best"], ck["best_state"], ck["best_smooth"], ck["bad"]
        start = ck["epoch"] + 1
        log(f"  resumed from {ckpt_path.name} after epoch {start - 1}")
    stopped = start > 1 and len(history) >= stop["min_epochs"] and bad >= stop["patience"]
    for epoch in range(start, stop["max_epochs"] + 1):
        if stopped:
            break
        t0 = time.time()
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Wtr)))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            X, Y, M = Wtr.get(idx)
            loss = masked_mse(net(X), Y, M)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            total += loss.item() * len(idx)
        net.eval()
        val_loss, val_cm = evaluate_split(net, Wva, norm)
        history.append(dict(epoch=epoch, train_loss=total / len(perm), val_loss=val_loss, val_err_1step_cm=val_cm,
                            seconds=time.time() - t0))
        if val_loss < best:      # weights of the best epoch are kept whatever the stopping rule
            best = val_loss
            best_state = {k: x.clone() for k, x in net.state_dict().items()}
        if val_loss < best_smooth * (1 - stop["min_rel_improvement"]):
            best_smooth, bad = val_loss, 0
        else:
            bad += 1
        stopped = epoch >= stop["min_epochs"] and bad >= stop["patience"]
        log(f"  epoch {epoch:2d}: train {total / len(perm):.5f}  val {val_loss:.5f}  "
            f"val 1-step error {val_cm:.4f} cm  ({time.time() - t0:.0f} s)  no improvement for {bad}")
        torch.save(dict(epoch=epoch, net=net.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict() if sched is not None else None, rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), history=history, best=best, best_state=best_state,
                        best_smooth=best_smooth, bad=bad), ckpt_path.with_suffix(".tmp"))
        ckpt_path.with_suffix(".tmp").replace(ckpt_path)    # atomic: a kill mid-save keeps the previous checkpoint
        (out / "progress.json").write_text(json.dumps({"epochs": len(history), "bad": bad, "stopped": stopped,
                                                       "max_epochs": stop["max_epochs"]}), encoding="utf-8")
    net.load_state_dict(best_state)
    best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
    save(out / "model.pt", net, norm, history)
    (out / "history.json").write_text(json.dumps({"variant": args.variant, "seed": seed, **v, "best_epoch": best_epoch,
                                                  "lr": args.lr, "warmup": args.warmup, "batch": args.batch,
                                                  "stopping": stop, "history": history}, indent=1), encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)
    (out / "progress.json").unlink(missing_ok=True)
    log(f"finished: best epoch {best_epoch}, val 1-step error {history[best_epoch - 1]['val_err_1step_cm']:.4f} cm")


if __name__ == "__main__":
    main()
