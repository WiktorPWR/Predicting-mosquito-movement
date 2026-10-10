"""Loading the trajectories and the fixed train/test split.

The test set is the frozen benchmark in Data/Benchmark/ (the 20 longest filtered
trajectories, made by Data/Processing/make_benchmark.py); every algorithm is
scored on exactly these. Learned models must train only on `train_set()`, which
is every other filtered trajectory.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data" / "Filtered_data" / "10-27-23-100-Mosquitoes-Empty-Chamber"
BENCHMARK_DIR = ROOT / "Data" / "Benchmark"
RESULTS_DIR = ROOT / "Results"

FRAME_S = 0.01   # one sample = 10 ms


@dataclass
class Trajectory:
    tid: int
    pos: np.ndarray       # (n, 3) cm, one row per frame, gaps already filled
    measured: np.ndarray  # (n,) bool, False for interpolated frames
    speed: np.ndarray     # (n,) m/s
    acc: np.ndarray       # (n,) m/s^2 (tangential, as in the raw data)

    @property
    def t(self):
        return np.arange(len(self.pos)) * FRAME_S


def load_trajectory(tid, data_dir=DATA_DIR):
    df = pd.read_csv(data_dir / f"traj_{tid:05d}.csv")
    return Trajectory(tid, df[["x_cm", "y_cm", "z_cm"]].to_numpy(), df["interpolated"].to_numpy() == 0,
                      df["speed_mps"].to_numpy(), df["acceleration_mps2"].to_numpy())


def summary(data_dir=DATA_DIR):
    return pd.read_csv(data_dir / "_summary.csv").sort_values("duration_s", ascending=False)


def test_ids():
    return summary(BENCHMARK_DIR)["trajectory_id"].tolist()


def test_set():
    return [load_trajectory(t, BENCHMARK_DIR) for t in test_ids()]


def train_set(data_dir=DATA_DIR):
    test = set(test_ids())
    return [load_trajectory(t, data_dir) for t in summary(data_dir)["trajectory_id"] if t not in test]


def max_speed(trajs, q=0.999):
    """Speed (cm/s) at quantile q over measured frames."""
    return 100 * np.quantile(np.concatenate([t.speed[t.measured] for t in trajs]), q)


def measurement_noise(trajs):
    """Per-axis position noise std (cm) from second differences of consecutive measured frames.

    For smooth motion sampled at 100 Hz, var(x[i+1] - 2x[i] + x[i-1]) ~ 6 sigma^2.
    """
    d2 = []
    for t in trajs:
        ok = t.measured[:-2] & t.measured[1:-1] & t.measured[2:]
        d2.append((t.pos[2:] - 2 * t.pos[1:-1] + t.pos[:-2])[ok])
    d2 = np.concatenate(d2)
    # robust std (MAD) so manoeuvres do not inflate the noise estimate
    mad = np.median(np.abs(d2 - np.median(d2, 0)), 0) * 1.4826
    return mad / np.sqrt(6)
