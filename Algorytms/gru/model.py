"""Small GRU networks for 3D mosquito trajectory prediction (PyTorch, CPU).

Four input representations over a window of the last `window` frames (default WINDOW = 20):
  "disp"  velocity only: frame-to-frame displacements (= velocity x 10 ms) / training std, no position
                                                                            -> input (batch, window, 3)
          output: the next displacement(s) (normalised); positions = last position + sum of displacements
  "pos"   positions (standardised with the training mean/std) and per-axis
          velocities (displacement / 10 ms, divided by the training std)    -> input (batch, window, 6)
          output: the next position (standardised)
  "posonly" positions only (standardised), no velocity                    -> input (batch, window, 3)
          output: the next position (standardised)
  "full"  positions + velocities (as "pos") + per-axis accelerations
          (change of displacement between frames / training std)           -> input (batch, window, 9)
          output: the next displacement (normalised, as "disp")

Network: nn.GRU(input, hidden, 1 layer, batch_first) -> nn.Linear(hidden, 3 * steps) on the last hidden
state. steps = 1: one frame ahead; steps = K ("disp" only): the next K displacements at once.
Multi-step forecast: the predicted frame(s) are appended to the window, the oldest dropped, and the network
is run again on the new window (sliding window, exactly the input the network was trained on).
"""
import numpy as np
import torch
from torch import nn

from Algorytms.common.data import FRAME_S

WINDOW = 20
POSITION_OUTPUT = ("pos", "posonly")   # modes whose network predicts the next absolute position
CHUNK = 10000   # origins forecast at once


class GRUNet(nn.Module):
    def __init__(self, n_in, hidden=32, steps=1):
        super().__init__()
        self.steps = steps
        self.gru = nn.GRU(n_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 3 * steps)

    def forward(self, x):
        """(batch, window, n_in) -> (batch, 3) for steps = 1, else (batch, steps, 3)."""
        out, _ = self.gru(x)
        y = self.head(out[:, -1])
        return y if self.steps == 1 else y.view(-1, self.steps, 3)


class Normaliser:
    """Statistics of the training split, stored with the model."""

    def __init__(self, disp_std, pos_mean, pos_std, acc_std=None):
        self.disp_std = np.asarray(disp_std, np.float32)
        self.pos_mean = np.asarray(pos_mean, np.float32)
        self.pos_std = np.asarray(pos_std, np.float32)
        self.acc_std = None if acc_std is None else np.asarray(acc_std, np.float32)  # std of displacement change
        self.vel_std = self.disp_std / FRAME_S

    def to_dict(self):
        keys = ("disp_std", "pos_mean", "pos_std") + (("acc_std",) if self.acc_std is not None else ())
        return {k: getattr(self, k).tolist() for k in keys}

    @staticmethod
    def history(mode, window=WINDOW):
        """Number of past positions (incl. the current one) needed for one input window."""
        return window + 2 if mode == "full" else window + 1

    def features(self, pos_win, mode):
        """pos_win: (n, history(mode, window), 3) positions [cm] -> network input (n, window, 3 / 6 / 9)."""
        d = np.diff(pos_win, axis=1)
        if mode == "disp":
            return d / self.disp_std
        if mode == "posonly":
            return (pos_win[:, 1:] - self.pos_mean) / self.pos_std
        if mode == "pos":
            return np.concatenate([(pos_win[:, 1:] - self.pos_mean) / self.pos_std, d / FRAME_S / self.vel_std], axis=2)
        a = np.diff(d, axis=1)
        return np.concatenate([(pos_win[:, 2:] - self.pos_mean) / self.pos_std, d[:, 1:] / self.disp_std,
                               a / self.acc_std], axis=2)

    def target(self, pos_last, pos_next, mode):
        if mode in POSITION_OUTPUT:
            return (pos_next - self.pos_mean) / self.pos_std
        return (pos_next - pos_last) / self.disp_std

    def decode(self, out, pos_last, mode):
        """Network output -> next position [cm]."""
        return out * self.pos_std + self.pos_mean if mode in POSITION_OUTPUT else pos_last + out * self.disp_std


class GRUPredictor:
    """Benchmark interface around a trained GRUNet."""

    def __init__(self, net, norm, mode, name, label, window=WINDOW):
        self.net, self.norm, self.mode, self.window = net.eval(), norm, mode, window
        self.steps = net.steps
        self.name, self.label = name, label
        self.params = dict(mode=mode, window=window, steps=self.steps, hidden=net.gru.hidden_size, **norm.to_dict())

    @torch.no_grad()
    def forecast(self, pos, measured, origins, max_h):
        out = np.empty((len(origins), max_h, 3), np.float32)
        pos = pos.astype(np.float32)
        n_hist = self.norm.history(self.mode, self.window)
        for a in range(0, len(origins), CHUNK):
            o = origins[a:a + CHUNK]
            # past positions; before the trajectory start the first position is repeated (zero displacement)
            win = pos[np.maximum(o[:, None] + np.arange(-n_hist + 1, 1), 0)]
            h = 0
            while h < max_h:
                x = torch.from_numpy(self.norm.features(win, self.mode).astype(np.float32))
                y = self.net(x).numpy()
                if self.steps == 1:
                    nxt = self.norm.decode(y, win[:, -1], self.mode)[:, None]
                else:                                   # K displacements -> K positions
                    nxt = win[:, -1][:, None] + np.cumsum(y * self.norm.disp_std, axis=1)
                k = min(len(nxt[0]), max_h - h)
                out[a:a + CHUNK, h:h + k] = nxt[:, :k]
                win = np.concatenate([win[:, nxt.shape[1]:], nxt], axis=1)
                h += k
        return out


def save(path, net, norm, mode, history, window=WINDOW):
    torch.save({"state": net.state_dict(), "n_in": net.gru.input_size, "hidden": net.gru.hidden_size,
                "steps": net.steps, "window": window, "norm": norm.to_dict(), "mode": mode, "history": history}, path)


def load(path, name, label=None):
    ck = torch.load(path, weights_only=False)
    net = GRUNet(ck["n_in"], ck["hidden"], ck.get("steps", 1))
    net.load_state_dict(ck["state"])
    window = ck.get("window", WINDOW)
    return GRUPredictor(net, Normaliser(**ck["norm"]), ck["mode"], name, label or default_label(ck), window)


INPUT_NAME = {"disp": "velocity input only", "pos": "position + velocity input", "posonly": "position input only",
              "full": "position + velocity + acceleration input"}


def default_label(ck):
    extra = ""
    if ck.get("window", WINDOW) != WINDOW:
        extra += f", window {ck['window']}"
    if ck.get("steps", 1) != 1:
        extra += f", {ck['steps']}-step output"
    return f"GRU ({INPUT_NAME[ck['mode']]}, hidden {ck['hidden']}{extra})"
