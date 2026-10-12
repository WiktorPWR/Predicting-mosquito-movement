"""Reference models for the learned predictors, on the same data and benchmark.

ConstantVelocity  the last displacement is repeated: p(t + h) = p(t) + h * (p(t) - p(t-1))
LinearAR          linear autoregression on displacements: the next displacement is a linear function of the
                  last WINDOW displacements of all three axes (3 * WINDOW inputs + bias -> 3 outputs),
                  fitted by least squares on the training split (one-step targets, measured frames only);
                  multi-step forecasts feed the prediction back (sliding window, as the GRU).
"""
import numpy as np

from Algorytms.common.ml_data import load_split, window_origins

WINDOW = 20


class ConstantVelocity:
    name, label = "constant_velocity", "Constant velocity (last displacement)"
    params = {}

    def forecast(self, pos, measured, origins, max_h):
        v = pos[origins] - pos[origins - 1]
        h = np.arange(1, max_h + 1)[None, :, None]
        return (pos[origins][:, None] + v[:, None] * h).astype(np.float32)


class LinearAR:
    name = "linear_ar"

    def __init__(self, window=WINDOW):
        self.window = window
        self.label = f"Linear autoregression ({window} displacements)"
        self.W = None

    def fit(self, split=None):
        split = split or load_split("train")
        t = window_origins(split, self.window, 1)
        X = split["disp"][t[:, None] + np.arange(-self.window + 1, 1)].reshape(len(t), -1).astype(np.float64)
        Y = split["disp"][t + 1].astype(np.float64)
        A = np.hstack([X, np.ones((len(X), 1))])
        self.W, *_ = np.linalg.lstsq(A, Y, rcond=None)
        self.params = dict(window=self.window, n_train_samples=len(t))
        return self

    def forecast(self, pos, measured, origins, max_h):
        d = np.diff(pos, axis=0, prepend=pos[:1])
        win = d[origins[:, None] + np.arange(-self.window + 1, 1)]         # (n, window, 3)
        p = pos[origins].astype(np.float64)
        out = np.empty((len(origins), max_h, 3), np.float32)
        for h in range(max_h):
            nxt = np.hstack([win.reshape(len(win), -1), np.ones((len(win), 1))]) @ self.W
            p = p + nxt
            out[:, h] = p
            win = np.concatenate([win[:, 1:], nxt[:, None]], axis=1)
        return out
