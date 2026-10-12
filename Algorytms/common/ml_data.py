"""Training windows for learned models, from Data/ML_dataset/ (made by Data/Processing/make_ml_dataset.py).

A sample at origin frame t (the last frame the model has seen):
    X = disp[t-window+1 .. t] / std      (window, 3)   normalised displacements, input
    Y = disp[t+1 .. t+horizon] / std     (horizon, 3)  normalised next displacements, target
    M = measured[t+1 .. t+horizon]       (horizon,)    True where the target frame is a real measurement
Kept when the whole window and target lie inside one trajectory, the origin frame is measured and at least
one target frame is measured. Interpolated frames may appear in the input (as in the benchmark) but must not
be learned as targets: use M to mask the loss. For horizon 1, M is all True.
"""
import json

import numpy as np

from Algorytms.common.data import ROOT

ML_DIR = ROOT / "Data" / "ML_dataset"


def load_split(name):
    """name: 'train', 'val' or 'test' -> dict of arrays (pos, disp, measured, offsets, ids)."""
    with np.load(ML_DIR / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def disp_std():
    return np.array(json.loads((ML_DIR / "norm.json").read_text(encoding="utf-8"))["disp_std_cm"], np.float32)


def window_origins(split, window=20, horizon=1, stride=1):
    """Origin frame indices (into the concatenated arrays) of all valid samples."""
    origins = []
    meas = split["measured"]
    for a, b in zip(split["offsets"][:-1], split["offsets"][1:]):
        # disp[a] is 0 (trajectory start), so the first usable displacement is a+1
        t = np.arange(a + window, b - horizon, stride)
        if len(t) == 0:
            continue
        any_target = np.zeros(len(t), bool)
        for h in range(1, horizon + 1):
            any_target |= meas[t + h]
        origins.append(t[meas[t] & any_target])
    return np.concatenate(origins) if origins else np.zeros(0, np.int64)


def make_windows(split, window=20, horizon=1, stride=1, std=None):
    """(X, Y, M, origins): X (n, window, 3), Y (n, horizon, 3) float32 normalised by the training std,
    M (n, horizon) bool target mask."""
    std = disp_std() if std is None else std
    t = window_origins(split, window, horizon, stride)
    d = split["disp"] / std
    X = d[t[:, None] + np.arange(-window + 1, 1)]
    Y = d[t[:, None] + np.arange(1, horizon + 1)]
    M = split["measured"][t[:, None] + np.arange(1, horizon + 1)]
    return X.astype(np.float32), Y.astype(np.float32), M, t
