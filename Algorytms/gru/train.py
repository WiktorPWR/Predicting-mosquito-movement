"""Train a GRU on the training split, early stopping on the validation split.

Adam (lr 1e-3), batch 64, MSE on the normalised target, one-step-ahead training. After every epoch the
validation loss and the validation one-step error in cm are recorded. The best epoch's weights (lowest
validation loss) are always kept.

Stopping rule per mode (STOPPING):
  default   at least 20 epochs, then stop when the validation loss has not improved for 5 epochs (max 40)
  pos, posonly   the validation loss of the position GRUs jumps strongly from epoch to epoch, so: at least
            20 epochs, then stop when the 3-epoch moving average of the validation loss has not improved by
            >= 1 % for 5 epochs (max 40 epochs)

With --steps K (velocity input only) the network outputs the next K displacements at once and the loss is
the MSE over the measured target frames (interpolated target frames are masked out).

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.gru.train <mode> [--hidden 32] [--window 20] [--steps 1]
        mode: disp (velocity only) | posonly | pos | full
Writes Results/gru/<mode>_h<hidden>[_w<window>][_k<steps>]/: model.pt, history.json, training_curve.png
(the suffixes only when window / steps differ from the defaults 20 / 1)
"""
import argparse
import json
import time

import numpy as np
import torch

from Algorytms.common.data import RESULTS_DIR
from Algorytms.common.ml_data import disp_std, load_split, window_origins
from Algorytms.common.plots import plot_training_curve
from Algorytms.gru.model import WINDOW, GRUNet, Normaliser, save

LR = 1e-3
BATCH = 64
SEED = 0
STOPPING = {
    "default": dict(max_epochs=40, min_epochs=20, patience=5, smooth=1, min_rel_improvement=0.0),
    "pos": dict(max_epochs=40, min_epochs=20, patience=5, smooth=3, min_rel_improvement=0.01),
}
STOPPING["posonly"] = STOPPING["pos"]


def acc_std(split):
    """Per-axis std of the change of displacement between consecutive frames (training split)."""
    dd = np.diff(split["disp"], axis=0)
    ok = np.ones(len(dd), bool)
    for a in split["offsets"][:-1]:
        ok[max(a - 1, 0):a + 1] = False      # differences touching a trajectory start
    return dd[ok].std(0)


def build(split, norm, mode, window=WINDOW, steps=1):
    """Inputs, targets and target masks for every valid sample of a split.

    X (n, window, features); Y (n, 3) for steps = 1, else (n, steps, 3) next displacements; M like Y without the
    last axis (True = measured target frame); plus the last input position and the true next position.
    """
    n_hist = norm.history(mode, window)
    t = window_origins(split, n_hist - 1, steps)
    win = split["pos"][t[:, None] + np.arange(-n_hist + 1, 1)]
    X = norm.features(win, mode).astype(np.float32)
    if steps == 1:
        Y = norm.target(win[:, -1], split["pos"][t + 1], mode)
        M = np.ones(len(t), bool)
    else:
        Y = split["disp"][t[:, None] + np.arange(1, steps + 1)] / norm.disp_std
        M = split["measured"][t[:, None] + np.arange(1, steps + 1)]
    return (torch.from_numpy(X), torch.from_numpy(Y.astype(np.float32)), torch.from_numpy(M),
            win[:, -1], split["pos"][t + 1])


def masked_mse(out, Y, M):
    m = M.unsqueeze(-1).float()
    return ((out - Y) ** 2 * m).sum() / (m.sum() * 3)


@torch.no_grad()
def evaluate_split(net, X, Y, M, last, true_next, norm, mode):
    out = torch.cat([net(X[i:i + 8192]) for i in range(0, len(X), 8192)])
    loss = float(masked_mse(out, Y, M))
    first = out.numpy() if out.dim() == 2 else out[:, 0].numpy()      # first predicted frame
    err = np.linalg.norm(norm.decode(first, last, mode) - true_next, axis=1)
    if M.dim() == 2:
        err = err[M[:, 0].numpy()]                                    # only measured next frames
    return loss, float(err.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["disp", "pos", "posonly", "full"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()
    mode, hidden, window, steps = args.mode, args.hidden, args.window, args.steps
    assert steps == 1 or mode == "disp", "multi-step output is implemented for the velocity-only GRU"
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out = RESULTS_DIR / "gru" / (f"{mode}_h{hidden}" + (f"_w{window}" if window != WINDOW else "")
                                 + (f"_k{steps}" if steps != 1 else ""))
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"     # state after the last finished epoch; deleted when training ends
    if (out / "model.pt").exists() and not ckpt_path.exists():
        print(f"{out.name}: already trained, nothing to do")
        return

    train, val = load_split("train"), load_split("val")
    norm = Normaliser(disp_std(), train["pos"].mean(0), train["pos"].std(0), acc_std(train))
    Xtr, Ytr, Mtr, _, _ = build(train, norm, mode, window, steps)
    Xva, Yva, Mva, last_va, next_va = build(val, norm, mode, window, steps)
    print(f"{out.name}: train {tuple(Xtr.shape)}, val {tuple(Xva.shape)}", flush=True)

    net = GRUNet(Xtr.shape[2], hidden, steps)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    stop = STOPPING.get(mode, STOPPING["default"])
    history, best, best_state, best_smooth, bad, start = [], np.inf, None, np.inf, 0, 1
    if ckpt_path.exists():                # resume an interrupted run exactly where it stopped
        ck = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
        history, best, best_state, best_smooth, bad = ck["history"], ck["best"], ck["best_state"], ck["best_smooth"], ck["bad"]
        start = len(history) + 1
        print(f"  resumed after epoch {len(history)}", flush=True)
    stopped = start > 1 and len(history) >= stop["min_epochs"] and bad >= stop["patience"]
    for epoch in range(start, stop["max_epochs"] + 1):
        if stopped:
            break
        t0 = time.time()
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = masked_mse(net(Xtr[idx]), Ytr[idx], Mtr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        val_loss, val_cm = evaluate_split(net, Xva, Yva, Mva, last_va, next_va, norm, mode)
        history.append(dict(epoch=epoch, train_loss=total / len(perm), val_loss=val_loss, val_err_1step_cm=val_cm,
                            seconds=time.time() - t0))
        print(f"  epoch {epoch:2d}: train {total / len(perm):.5f}  val {val_loss:.5f}  "
              f"val 1-step error {val_cm:.3f} cm  ({time.time() - t0:.0f} s)", flush=True)
        if val_loss < best:      # weights of the best epoch are kept whatever the stopping rule
            best = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        smooth = np.mean([h["val_loss"] for h in history[-stop["smooth"]:]])
        if smooth < best_smooth * (1 - stop["min_rel_improvement"]):
            best_smooth, bad = smooth, 0
        else:
            bad += 1
        stopped = epoch >= stop["min_epochs"] and bad >= stop["patience"]
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), history=history, best=best, best_state=best_state,
                        best_smooth=best_smooth, bad=bad), ckpt_path.with_suffix(".tmp"))
        ckpt_path.with_suffix(".tmp").replace(ckpt_path)    # atomic: a kill mid-save keeps the previous checkpoint
    net.load_state_dict(best_state)
    best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
    save(out / "model.pt", net, norm, mode, history, window)
    (out / "history.json").write_text(json.dumps({"best_epoch": best_epoch, "lr": LR, "batch": BATCH,
                                                  "stopping": stop, "window": window, "steps": steps,
                                                  "hidden": hidden,
                                                  "history": history}, indent=1), encoding="utf-8")
    plot_training_curve(history, best_epoch, f"GRU {out.name}: training",
                        out / "training_curve.png")
    ckpt_path.unlink(missing_ok=True)
    print(f"best epoch {best_epoch}, val 1-step error {history[best_epoch - 1]['val_err_1step_cm']:.3f} cm")


if __name__ == "__main__":
    main()
