# Kalman filter (constant velocity)

**Final model: Kalman CV, setting 0.336 m/s.** Results: [Results/kalman/](../../Results/kalman/).

| file | purpose |
|---|---|
| `kalman_filter.py` | the filter (`KalmanCV`), chosen setting `SETTING_CM_S = 33.6` |
| `run.py` | evaluates the chosen setting → `Results/kalman/cv_0.336mps/` (~40 s) |
| `sweep.py` | setting sweep → `Results/kalman/sweep/` (~7 min; `--plot` redraws from the saved CSV) |

## Model

State `[x, y, z, vx, vy, vz]`, constant velocity with white-noise acceleration, axes independent.
One tuning value, the **velocity setting s**:

- initial velocity std = s (the direction is unknown at the start);
- process noise q = s² / 1 s: the velocity may drift by ~s within one second;
- measurement noise from the training data (robust second differences): 0.0076 / 0.0072 / 0.0512 cm in x / y / z;
- interpolated frames are used as measurements with 1 cm noise (the measured gap-filling error), so the filter
  never runs blind through a gap.

The forecast is the constant-velocity extrapolation of the filtered state: p(t + h) = p(t) + v(t)·h.

The initial setting was half the maximum speed: v_max = 1.679 m/s (99.9th percentile of the measured speed in the
training data; the absolute maximum, 2.96 m/s, is most likely measurement error) → s = 0.84 m/s.

## Tests

### 1. Constant velocity vs constant acceleration (setting 0.84 m/s)

| horizon | CV mean | CA mean | CV max | CA max |
|---|---|---|---|---|
| 0.01 s | 0.16 cm | 0.28 cm | 1.3 cm | 2.3 cm |
| 0.1 s | 1.84 cm | 3.10 cm | 14 cm | 23 cm |
| 0.5 s | 13.6 cm | 47.1 cm | 80 cm | 3.1 m |
| 1 s | 34.6 cm | 1.76 m | 1.65 m | 11.5 m |
| 5 s | 2.5 m | 41.4 m | 10 m | 274 m |

The CA model estimates acceleration from noise and its parabola leaves the chamber; worse at every horizon.
**CA was dropped.**

### 2. Velocity setting sweep

10 settings evenly spaced over (0, v_max]: 0.168, 0.336, …, 1.679 m/s (0 itself makes the filter stop following
the measurements). Mean error [cm] on the benchmark:

| setting [m/s] | 0.01 s | 0.1 s | 0.5 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|---|---|
| 0.168 | 0.19 | **1.71** | **12.7** | **32.8** | **82.0** | **244** |
| **0.336** | 0.17 | 1.76 | 13.1 | 33.6 | 83.5 | 248 |
| 0.840 | 0.16 | 1.84 | 13.6 | 34.6 | 85.3 | 252 |
| 1.007 | **0.15** | 1.86 | 13.7 | 34.8 | 85.7 | 252 |
| 1.679 | 0.16 | 1.92 | 14.1 | 35.4 | 86.7 | 254 |

Full table: [sweep_mean_error.md](../../Results/kalman/sweep/sweep_mean_error.md).

- Smaller settings smooth the velocity more and win from ~0.05 s on; larger ones win only 1–2 samples ahead.
- The setting matters little: the worst is only 4–7 % worse than the best at most horizons (max. 15 % at ~0.15 s).
- **0.336 m/s** has the lowest mean ratio to the best setting over 12 horizons (0.01–5 s), practically tied with
  0.168 m/s (which is 20 % worse one sample ahead) → chosen.

## Results (setting 0.336 m/s, 20 benchmark trajectories, ~72 000 predictions per horizon)

| horizon | mean | median | p95 | max | within 1 cm | within 5 cm | within 10 cm |
|---|---|---|---|---|---|---|---|
| 0.01 s | 0.17 cm | 0.12 cm | 0.47 cm | 1.35 cm | 99.9 % | 100 % | 100 % |
| 0.05 s | 0.84 cm | 0.68 cm | 2.08 cm | 6.4 cm | 68.8 % | 100 % | 100 % |
| 0.1 s | 1.76 cm | 1.50 cm | 4.01 cm | 13.2 cm | 29.7 % | 98.2 % | 100 % |
| 0.2 s | 3.98 cm | 3.43 cm | 8.90 cm | 29.1 cm | 6.1 % | 71.7 % | 97.2 % |
| 0.5 s | 13.1 cm | 11.1 cm | 30.4 cm | 73 cm | 0.3 % | 15.1 % | 44.0 % |
| 1 s | 33.6 cm | 28.7 cm | 78.0 cm | 1.56 m | 0 % | 2.1 % | 9.8 % |
| 2 s | 83.5 cm | 73.5 cm | 1.84 m | 3.29 m | 0 % | 0.2 % | 1.4 % |
| 5 s | 2.48 m | 2.35 m | 4.99 m | 9.40 m | 0 % | 0 % | 0.1 % |

Full table with every metric and per-trajectory means: [summary.md](../../Results/kalman/cv_0.336mps/summary.md).

## Conclusions

- Up to ~0.1 s the constant-velocity Kalman filter is accurate (< 2 cm mean, 98 % within 5 cm).
- Beyond ~0.5 s no constant-velocity model can follow: the mosquito turns more than once per second, and no setting
  brings the 1 s error below ~33 cm.
- One sample ahead, plain 2-point extrapolation (0.12 cm) is slightly better than the filter (0.17 cm): the data
  are already smoothed by the tracker, so the filter's extra smoothing costs a little at the shortest horizon.
- Convergence after a cold start ([Results/particle/convergence/](../../Results/particle/convergence/)): the
  filter is **sensible from the 2nd sample** (0.02 s) at every horizon: after one sample it knows no velocity
  (0.1 s error 4.9 cm), after two it is already at its steady state (1.7 cm).
