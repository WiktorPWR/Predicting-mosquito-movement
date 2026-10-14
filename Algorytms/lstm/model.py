"""LSTM vs GRU study: recurrent networks predicting the next 10 displacements at once (PyTorch, CPU).

Every variant is the selected GRU (Results/gru/disp_h32_k10) with one thing changed:
  cell     "gru" | "lstm"
  inp      "disp"     input = the last `window` frame-to-frame displacements / training std     (window, 3)
           "dispacc"  as "disp" plus accelerations a_t = d_t - d_(t-1) / training std per axis (window, 6)
  hidden   size of the hidden state
  window   number of input frames (10 ms each)
Output (all variants): Linear(hidden -> 30) on the last hidden state -> the next 10 displacements (10, 3),
normalised by the training std; positions = last position + cumulative sum of the decoded displacements.
Forecasts beyond 0.1 s: the 10 predicted positions are appended to the input window, the 10 oldest dropped,
and the network is run again; accelerations of the new window are computed from the predicted positions.

LSTM biases: all zero except the forget gate, bias_ih + bias_hh = 0.5 + 0.5 = 1 (PyTorch gate order:
input, forget, cell, output).
"""
import io

import numpy as np
import torch
from torch import nn

STEPS = 10
CHUNK = 10000   # origins forecast at once

# the variants of the study; label is used in tables and figures
VARIANTS = {
    "gru_base":  dict(cell="gru",  inp="disp",    hidden=32, window=20,  label="GRU baseline"),
    "lstm_a":    dict(cell="lstm", inp="disp",    hidden=32, window=20,  label="LSTM A (H32)"),
    "lstm_a28":  dict(cell="lstm", inp="disp",    hidden=28, window=20,  label="LSTM A' (H28)"),
    "lstm_b":    dict(cell="lstm", inp="dispacc", hidden=32, window=20,  label="LSTM B (input 6)"),
    "gru_b":     dict(cell="gru",  inp="dispacc", hidden=32, window=20,  label="GRU-B (input 6)"),
    "lstm_c50":  dict(cell="lstm", inp="disp",    hidden=32, window=50,  label="LSTM C (window 50)"),
    "lstm_c100": dict(cell="lstm", inp="disp",    hidden=32, window=100, label="LSTM C (window 100)"),
}
SEEDS = range(5)
N_IN = {"disp": 3, "dispacc": 6}


class RNNNet(nn.Module):
    def __init__(self, cell, n_in, hidden=32, steps=STEPS, layers=1):
        super().__init__()
        self.cell, self.steps = cell, steps
        self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 3 * steps)
        if cell == "lstm":
            for name, p in self.rnn.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(p)
                    p.data[hidden:2 * hidden] = 0.5      # forget gate (every layer): bias_ih + bias_hh = 1

    def forward(self, x):
        """(batch, window, n_in) -> (batch, steps, 3)."""
        _, h = self.rnn(x)
        h = h[0] if self.cell == "lstm" else h            # LSTM returns (h, c)
        return self.head(h[-1]).view(-1, self.steps, 3)       # h[-1]: last layer


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def weights_bytes(net):
    """Size of the saved weights alone (state_dict), comparable between variants."""
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.tell()


class Normaliser:
    """Per-axis statistics of the training split, stored with the model."""

    def __init__(self, disp_std, acc_std=None):
        self.disp_std = np.asarray(disp_std, np.float32)
        self.acc_std = None if acc_std is None else np.asarray(acc_std, np.float32)

    def to_dict(self):
        return {"disp_std": self.disp_std.tolist(),
                **({"acc_std": self.acc_std.tolist()} if self.acc_std is not None else {})}

    @staticmethod
    def history(inp, window):
        """Number of past positions (incl. the current one) needed for one input window."""
        return window + 2 if inp == "dispacc" else window + 1

    def frame_features(self, disp, inp):
        """Per-frame network input from displacements (n, 3) -> (n, 3 / 6); row t uses d_t (and d_(t-1))."""
        d = disp / self.disp_std
        if inp == "disp":
            return d
        a = np.zeros_like(disp)
        a[1:] = (disp[1:] - disp[:-1]) / self.acc_std
        return np.concatenate([d, a], axis=1)

    def features(self, pos_win, inp):
        """pos_win: (n, history, 3) positions [cm] -> network input (n, window, 3 / 6)."""
        d = np.diff(pos_win, axis=1)
        if inp == "disp":
            return d / self.disp_std
        return np.concatenate([d[:, 1:] / self.disp_std, np.diff(d, axis=1) / self.acc_std], axis=2)


class Predictor:
    """Benchmark interface (forecast(pos, measured, origins, max_h)) around a trained RNNNet."""

    def __init__(self, net, norm, inp, window, name, label):
        self.net, self.norm, self.inp, self.window = net.eval(), norm, inp, window
        self.name, self.label = name, label
        self.params = dict(cell=net.cell, inp=inp, window=window, hidden=net.rnn.hidden_size,
                           layers=net.rnn.num_layers, steps=net.steps,
                           **norm.to_dict())

    @torch.no_grad()
    def forecast(self, pos, measured, origins, max_h):
        out = np.empty((len(origins), max_h, 3), np.float32)
        pos = pos.astype(np.float32)
        n_hist = self.norm.history(self.inp, self.window)
        for a in range(0, len(origins), CHUNK):
            o = origins[a:a + CHUNK]
            # past positions; before the trajectory start the first position is repeated (zero displacement)
            win = pos[np.maximum(o[:, None] + np.arange(-n_hist + 1, 1), 0)]
            h = 0
            while h < max_h:
                y = self.net(torch.from_numpy(self.norm.features(win, self.inp).astype(np.float32))).numpy()
                nxt = win[:, -1][:, None] + np.cumsum(y * self.norm.disp_std, axis=1)
                k = min(nxt.shape[1], max_h - h)
                out[a:a + CHUNK, h:h + k] = nxt[:, :k]
                win = np.concatenate([win[:, nxt.shape[1]:], nxt], axis=1)
                h += k
        return out


def save(path, net, norm, inp, window, history):
    torch.save({"state": net.state_dict(), "cell": net.cell, "n_in": net.rnn.input_size,
                "hidden": net.rnn.hidden_size, "layers": net.rnn.num_layers, "steps": net.steps, "window": window, "inp": inp,
                "norm": norm.to_dict(), "history": history}, path)


def load(path, name, label):
    ck = torch.load(path, weights_only=False)
    if "cell" not in ck:          # model saved by Algorytms/gru/train.py (GRU baseline, seed 0)
        ck = dict(ck, cell="gru", inp="disp", norm={"disp_std": ck["norm"]["disp_std"]},
                  state={k.replace("gru.", "rnn."): v for k, v in ck["state"].items()})
    net = RNNNet(ck["cell"], ck["n_in"], ck["hidden"], ck["steps"], ck.get("layers", 1))
    net.load_state_dict(ck["state"])
    return Predictor(net, Normaliser(**ck["norm"]), ck["inp"], ck["window"], name, label)
