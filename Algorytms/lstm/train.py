"""Train one variant of the LSTM vs GRU study with one seed (CPU only).

Training is identical to the selected GRU (Algorytms/gru/train.py, disp --steps 10): same split, same
normalisation, Adam lr 1e-3, batch 64, MSE over the measured target frames (interpolated target frames masked
out), at least 20 epochs, then stop after 5 epochs without improvement of the validation loss (max 40), best
epoch's weights kept. The constants and the loss are imported from there. Training windows are gathered from a
per-frame feature table batch by batch (the same numbers as materialising them; a few hundred MB per run even
for window 100). With seed 0 the GRU baseline reproduces Results/gru/disp_h32_k10.

Usage (from the repo root):
    python -m Algorytms.lstm.train <variant> --seed <k> [--threads 1]
        variant: gru_base | lstm_a | lstm_a28 | lstm_b | gru_b | lstm_c50 | lstm_c100   (see model.VARIANTS)
    Optional overrides (defaults = the study; any change goes to its own folder, outside the study table):
        --hidden H  --layers L  --batch B  --max-epochs E  --checkpoint PATH
Writes Results/lstm/<variant>[_h<H>][_l<L>][_b<B>][_e<E>]/seed<k>/: model.pt, history.json, train.log
(progress.json while training). A per-epoch checkpoint (model + optimiser + epoch + random state) lets an
interrupted run continue exactly where it stopped: run the same command again.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from Algorytms.common.data import RESULTS_DIR
from Algorytms.common.ml_data import disp_std, load_split, window_origins
from Algorytms.gru.train import BATCH, LR, STOPPING, acc_std, masked_mse
from Algorytms.lstm.model import N_IN, STEPS, VARIANTS, Normaliser, RNNNet, save

OUT_DIR = RESULTS_DIR / "lstm"


class Windows:
    """All valid samples of a split: origin frames t; inputs, targets and masks gathered on demand."""

    def __init__(self, split, norm, inp, window):
        self.t = torch.from_numpy(window_origins(split, norm.history(inp, window) - 1, STEPS))
        self.F = torch.from_numpy(norm.frame_features(split["disp"], inp).astype(np.float32))
        self.D = torch.from_numpy((split["disp"] / norm.disp_std).astype(np.float32))
        self.meas = torch.from_numpy(split["measured"])
        self.in_offs = torch.arange(-window + 1, 1)
        self.out_offs = torch.arange(1, STEPS + 1)

    def __len__(self):
        return len(self.t)

    def get(self, idx):
        t = self.t[idx][:, None]
        return self.F[t + self.in_offs], self.D[t + self.out_offs], self.meas[t + self.out_offs]


@torch.no_grad()
def evaluate_split(net, W, norm):
    """Validation loss and the mean error [cm] of the first predicted frame (measured next frames only)."""
    loss_sum, w_sum, err = 0.0, 0.0, []
    for a in range(0, len(W), 8192):
        X, Y, M = W.get(torch.arange(a, min(a + 8192, len(W))))
        out = net(X)
        m = M.float().sum().item() * 3
        loss_sum += masked_mse(out, Y, M).item() * m
        w_sum += m
        e = np.linalg.norm((out[:, 0] - Y[:, 0]).numpy() * norm.disp_std, axis=1)
        err.append(e[M[:, 0].numpy()])
    return loss_sum / w_sum, float(np.concatenate(err).mean())


def run_dir(variant, seed, hidden, layers, batch, max_epochs):
    v = VARIANTS[variant]
    suffix = "".join(f"_{k}{x}" for k, x, d in (("h", hidden, v["hidden"]), ("l", layers, 1), ("b", batch, BATCH),
                                                 ("e", max_epochs, STOPPING["default"]["max_epochs"])) if x != d)
    return OUT_DIR / (variant + suffix) / f"seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("variant", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 when several runs go in parallel)")
    ap.add_argument("--hidden", type=int, help="hidden size (default: the variant's)")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-epochs", type=int, default=STOPPING["default"]["max_epochs"])
    ap.add_argument("--checkpoint", help="checkpoint file (default: checkpoint.pt in the run folder)")
    args = ap.parse_args()
    v, seed = VARIANTS[args.variant], args.seed
    hidden = args.hidden or v["hidden"]
    stop = dict(STOPPING["default"], max_epochs=args.max_epochs,
                min_epochs=min(STOPPING["default"]["min_epochs"], args.max_epochs))
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = run_dir(args.variant, seed, hidden, args.layers, args.batch, args.max_epochs)
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
    norm = Normaliser(disp_std(), acc_std(train) if v["inp"] == "dispacc" else None)
    Wtr, Wva = Windows(train, norm, v["inp"], v["window"]), Windows(val, norm, v["inp"], v["window"])
    net = RNNNet(v["cell"], N_IN[v["inp"]], hidden, STEPS, args.layers)
    log(f"{out.parent.name} seed {seed}: {v['cell']} input {v['inp']} hidden {hidden} layers {args.layers} "
        f"window {v['window']} batch {args.batch}, {sum(p.numel() for p in net.parameters())} parameters, "
        f"{args.threads} thread(s); train {len(Wtr)} windows, val {len(Wva)}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    history, best, best_state, best_smooth, bad, start = [], np.inf, None, np.inf, 0, 1
    if ckpt_path.exists():                # resume an interrupted run exactly where it stopped
        ck = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
        history, best, best_state, best_smooth, bad = ck["history"], ck["best"], ck["best_state"], ck["best_smooth"], ck["bad"]
        start = ck.get("epoch", len(history)) + 1
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
        torch.save(dict(epoch=epoch, net=net.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), history=history, best=best, best_state=best_state,
                        best_smooth=best_smooth, bad=bad), ckpt_path.with_suffix(".tmp"))
        ckpt_path.with_suffix(".tmp").replace(ckpt_path)    # atomic: a kill mid-save keeps the previous checkpoint
        (out / "progress.json").write_text(json.dumps({"epochs": len(history), "bad": bad, "stopped": stopped,
                                                       "max_epochs": stop["max_epochs"]}), encoding="utf-8")
    net.load_state_dict(best_state)
    best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
    save(out / "model.pt", net, norm, v["inp"], v["window"], history)
    (out / "history.json").write_text(json.dumps({"variant": args.variant, "seed": seed, **v, "hidden": hidden,
                                                  "layers": args.layers, "best_epoch": best_epoch, "lr": LR,
                                                  "batch": args.batch, "stopping": stop,
                                                  "history": history}, indent=1), encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)
    (out / "progress.json").unlink(missing_ok=True)
    log(f"finished: best epoch {best_epoch}, val 1-step error {history[best_epoch - 1]['val_err_1step_cm']:.4f} cm")


if __name__ == "__main__":
    main()
