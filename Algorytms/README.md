# Algorithms

Code only; results go to [../Results/](../Results/). Run everything from the repository root with
`.venv/Scripts/python -m Algorytms.<module>`.

| folder | content |
|---|---|
| [common/](common/) | shared data loading, training windows (`ml_data.py`), evaluation protocol, figures, 3D viewer, convergence analysis |
| [kalman/](kalman/README.md) | Kalman filter (constant velocity) |
| [particle/](particle/README.md) | particle filter |
| [baselines/](baselines/README.md) | constant velocity, linear autoregression |
| [gru/](gru/README.md) | GRU networks (displacement / position input) |
| `compare_models.py` | one table and figure comparing all evaluated algorithms → `Results/comparison/` |

## Evaluation protocol (same for every algorithm)

Implemented in [common/evaluation.py](common/evaluation.py).

- **Test set:** the 20 trajectories in `Data/Benchmark/` (39–106 s each, 17.2 min). Parameters that need data
  statistics (maximum speed, measurement noise) and anything learned come only from the other 2 365 trajectories.
- **Input:** the gap-filled trajectory (one sample every 10 ms) with a flag marking interpolated frames.
- **Prediction origins:** every *measured* frame after 1 s of history (~72 000 origins). From each origin the
  model predicts the position 1 … 500 samples ahead (**10 ms … 5 s**) using only frames up to the origin.
- **Error:** Euclidean distance [cm] between predicted and true position. Only measured targets are scored;
  interpolated frames are never used as ground truth.
- **Statistics per horizon:** mean, median, RMSE, 25/75/90/95th percentile, maximum, and the share of predictions
  within 1 / 5 / 10 cm, pooled over all origins of all 20 trajectories.

Why origins only at measured frames: in the first Kalman run a "1-sample" prediction made right after a 1 s
interpolated gap was really a 1 s prediction (maximum 1-step error 95 cm). Scoring only predictions made right after
a real measurement keeps the horizon honest for every algorithm.

### Outputs of `evaluate()` (per algorithm / configuration)

| file | content |
|---|---|
| `summary.md` | table per horizon + mean error per trajectory |
| `error_vs_horizon.png` | mean, median, 25–75 % band, p95 and maximum vs horizon |
| `horizon_stats.csv` | all 500 horizons (used by `compare_models.py`) |
| `params.json` | parameters, display name, test ids |
| `viewer_data/` | data for the interactive 3D viewer |

### Convergence: from which sample on are predictions sensible?

[common/convergence.py](common/convergence.py): every model is started from scratch at every second of every
test trajectory (~900 cold starts) and fed only the frames from that point on. The error at horizons 0.01 / 0.1 /
0.5 / 1 s is averaged per number of samples seen since the start; the model is "sensible from sample k" once its
error stays within 15 % of the steady state (mean over samples 201–300). Result for both filters: from the 2nd sample.

### 3D viewer

`python -m Algorytms.common.viewer` → `Results/viewer/index.html` (works offline). For every test trajectory:
3D trajectory with the last second, the true future and every algorithm's prediction for a chosen horizon; time
slider / playback; error-over-time and speed charts (click to jump); for particle filters the particle cloud.

## Adding an algorithm

A model is any object with `name`, `label` and

```python
forecast(pos, measured, origins, max_h) -> array (len(origins), max_h, 3)
```

returning, for every origin frame `t`, the predicted positions at `t+1 … t+max_h` from frames `0 … t` only.
Then `evaluate(model, data.test_set(), out=RESULTS_DIR / "<algorithm>" / "<variant>")`; the comparison and the
viewer pick it up automatically.
