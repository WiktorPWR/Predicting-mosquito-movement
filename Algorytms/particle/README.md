# Particle filter

Results: [Results/particle/](../../Results/particle/).

| file | purpose |
|---|---|
| `particle_filter.py` | the filter (`ParticleFilter`) |
| `sweep.py` | `noise` sweep (process noise × likelihood width), `grid` sweep (particles × elite fraction), `plot` |
| `run.py` | full evaluation of the best configuration + convergence analysis (PF vs Kalman) |

## Model

Each particle is a state `[x, y, z, vx, vy, vz]` (3 positions and 3 velocities, every axis independent).
For every frame:

1. **predict:** constant velocity + white-noise acceleration, the same continuous model as the Kalman filter
   (q = setting² / 1 s), sampled exactly for one 10 ms step;
2. **weight:** Gaussian likelihood of the measured position with std = *likelihood sigma*
   (interpolated frames: max(1 cm, likelihood sigma));
3. **best particle:** the particle with the largest weight; **the forecast is its constant-velocity
   extrapolation** (it is drawn red in the viewer);
4. **elitist resampling:** keep the `keep` fraction of particles with the largest weights, throw the rest away and
   draw all N new particles from the kept ones (systematic resampling, proportional to weight).
   `keep = 100 %` is the classic bootstrap particle filter.

Start: all particles at the first measurement, velocities ~ N(0, setting²) per axis (as the Kalman filter's P0).
Runtime grows linearly with N: ~1 s (N = 100) to ~10 s (N = 5000) for a 53 s trajectory.

## Why the likelihood width is a parameter

With the real measurement noise (~0.008 cm in x/y, 0.05 cm in z) the weights are so peaked that every frame one
particle takes practically all the weight. The filter cannot move particles towards the measurement (unlike the
Kalman gain); it can only select. With one survivor per frame the velocities are never averaged, the cloud lags
behind manoeuvres and the best particle's velocity is noise. On trajectory 433 (N = 1000, keep 100 %):

| setting | likelihood sigma | mean error 0.01 s | 0.1 s | 0.5 s | 1 s |
|---|---|---|---|---|---|
| 0.336 m/s | real noise | 5.81 cm | 7.56 cm | 25.1 cm | 53.1 cm |
| 0.336 m/s | 0.5 cm | 0.34 cm | 2.06 cm | 12.6 cm | 31.7 cm |
| (Kalman 0.336 m/s) | | 0.23 cm | 2.17 cm | — | 34.2 cm |

A larger process noise alone helps one sample ahead but makes the velocities noisy (worse at longer horizons).
Hence the noise sweep covers both the process noise and the likelihood width.

## Visualisation

`python -m Algorytms.common.viewer` → `Results/viewer/index.html`. For the particle filter the main 3D view shows
the **predicted particle cloud** for the chosen horizon (every particle propagated with the motion model and its
process noise) and the **best particle in red** (the source of the prediction). A separate zoomed panel shows the
**current posterior cloud** around the measurement in mm: grey particles sized by weight, the most probable particle
in red with its velocity, the measurement as a black cross.

## Tests

1. **Noise sweep** (`sweep.py noise`): process-noise setting 0.168 … 1.679 m/s (10 values, as for the Kalman filter)
   × likelihood sigma 0.05 / 0.1 / 0.2 / 0.5 / 1 / 2 cm; N = 1000, keep 100 %. 60 runs on the benchmark.
2. **Particles × elite fraction** (`sweep.py grid`), with the best noise pair: N = 100, 250, 500, 1000, 2500, 5000 ×
   keep = 1, 5, 10, 25, 50, 100 %. 36 runs, runtime recorded.
3. **Full evaluation** of the best configuration on the benchmark (same protocol as every algorithm, see
   [../README.md](../README.md)).
4. **Convergence:** from which sample after a cold start are the predictions sensible (PF vs Kalman).

Selection criterion in every sweep: lowest mean ratio of the mean error to the best configuration's mean error over
12 horizons (0.01 … 5 s).

## Results

### 1. Noise sweep

Full table: [noise.md](../../Results/particle/sweep/noise.md); heatmaps `noise_heatmap_0.1s.png`, `noise_heatmap_1s.png`.
Best five of 60 (N = 1000, keep 100 %), mean error [cm]:

| setting [m/s] | likelihood sigma [cm] | 0.01 s | 0.1 s | 0.5 s | 1 s | 5 s | error / best |
|---|---|---|---|---|---|---|---|
| **0.504** | **0.2** | 0.25 | 2.08 | 14.0 | 34.8 | 250 | **1.027** |
| 0.504 | 0.5 | 0.25 | 2.12 | 14.1 | 34.8 | 249 | 1.035 |
| 0.336 | 0.2 | 0.33 | **1.97** | 13.5 | 33.9 | 247 | 1.048 |
| 0.336 | 0.5 | 0.33 | 2.00 | **13.4** | **33.7** | **246** | 1.050 |
| 0.672 | 0.2 | **0.24** | 2.22 | 14.6 | 35.9 | 253 | 1.056 |
| *Kalman CV 0.336* | | *0.17* | *1.76* | *13.1* | *33.6* | *248* | |

- **0.168 m/s fails completely** (12–29 cm already at 0.1 s): with so little process noise the cloud cannot follow
  the mosquito's manoeuvres and loses it.
- Too narrow a likelihood (0.05 cm) and too wide (2 cm) are both worse; **0.2–0.5 cm** is the sweet spot.
- Lower process noise wins at long horizons (smoother velocity), higher at the shortest; the selection criterion
  picks **0.504 m/s / 0.2 cm**, practically tied with 0.336 m/s / 0.2–0.5 cm.
- With 1000 particles the particle filter is **slightly worse than the Kalman filter at every horizon** — expected
  for a linear Gaussian motion model, for which the Kalman filter is optimal; the particle filter only approximates it.

### 2. Number of particles × elite fraction

Setting 0.504 m/s, likelihood sigma 0.2 cm. Full table: [grid.md](../../Results/particle/sweep/grid.md);
figures `grid_particles_effect.png` (error vs N per elite fraction), `grid_heatmap_*.png`, `grid_runtime.png`.

Mean error [cm], classic resampling (keep 100 %), and runtime on the whole benchmark (one process):

| particles | 0.01 s | 0.1 s | 1 s | 5 s | runtime |
|---|---|---|---|---|---|
| 100 | 0.42 | 2.23 | 35.6 | 253 | 28 s |
| 250 | 0.32 | 2.14 | 35.1 | 250 | 30 s |
| 500 | 0.28 | 2.10 | 34.9 | 250 | 37 s |
| 1000 | 0.25 | 2.08 | 34.8 | 250 | 59 s |
| 2500 | 0.23 | 2.06 | 34.7 | 249 | 108 s |
| **5000** | **0.22** | **2.05** | **34.6** | **249** | 172 s |

Elite fraction at N = 1000, mean error at 0.1 s: keep 1 % 2.36 cm, 5 % 2.21, 10 % 2.17, 25 % 2.12, 50 % 2.10,
100 % **2.08 cm**.

- **More particles always help, with diminishing returns.** Most of the gain is between 100 and 500 particles;
  from 1000 to 5000 the 0.1 s error drops only 1.6 % for 3× the runtime. One sample ahead the gain is larger
  (0.42 → 0.22 cm), because there the best particle's position precision matters most.
- **Keeping only the best particles never helps**: at every N the classic filter (100 %) is best, and keeping
  1 % is clearly worst (with 100 particles that is a single particle: 6.3 cm at 0.1 s). Throwing particles away
  removes the diversity of velocities the filter needs; the resampling already favours the best particles in
  proportion to their weights.
- **Best configuration: N = 5000, keep 100 %.** If runtime matters, N = 500–1000 is within 1–2 %.

### 3. Best configuration on the benchmark

**Particle filter: N = 5000, keep 100 %, process noise 0.504 m/s, likelihood sigma 0.2 cm.**
Full table: [summary.md](../../Results/particle/best/summary.md); comparison with the Kalman filter:
[comparison_table.md](../../Results/comparison/comparison_table.md).

| horizon | PF mean | PF median | PF p95 | PF max | within 5 cm | Kalman CV mean |
|---|---|---|---|---|---|---|
| 0.01 s | 0.22 cm | 0.17 cm | 0.51 cm | 6.9 cm | 100.0 % | 0.17 cm |
| 0.05 s | 1.02 cm | 0.89 cm | 2.18 cm | 8.8 cm | 99.9 % | 0.84 cm |
| 0.1 s | 2.05 cm | 1.81 cm | 4.22 cm | 15.4 cm | 97.7 % | 1.76 cm |
| 0.2 s | 4.45 cm | 3.92 cm | 9.34 cm | 33.2 cm | 66.8 % | 3.98 cm |
| 0.5 s | 13.9 cm | 11.8 cm | 31.1 cm | 84 cm | 9.6 % | 13.1 cm |
| 1 s | 34.6 cm | 29.5 cm | 78.3 cm | 1.71 m | 1.0 % | 33.6 cm |
| 2 s | 84.5 cm | 74.2 cm | 1.83 m | 3.98 m | 0.1 % | 83.5 cm |
| 5 s | 2.49 m | 2.34 m | 4.94 m | 9.40 m | 0 % | 2.48 m |

The particle filter is **close to but slightly worse than the Kalman filter at every horizon** (+16–30 % up to
0.2 s, +3–6 % from 0.5 s on, the same at 5 s), and ~40× slower. Its maximum one-sample error (6.9 cm vs 1.35 cm)
shows occasional moments where the cloud lags behind a sharp manoeuvre.
(The 0.336 m/s / 0.2 cm noise pair from the noise sweep is slightly better from 0.1 s on — 1.97 cm at 0.1 s —
but worse one sample ahead; the selection criterion weighs all 12 horizons equally.)

### 4. Convergence: from which sample on are predictions sensible?

[convergence.md](../../Results/particle/convergence/convergence.md), figure `convergence.png`. Both filters were
started from scratch 961 times (every second of every benchmark trajectory):

| horizon | error after 1 sample | after 2 samples (PF / Kalman) | steady state (PF / Kalman) |
|---|---|---|---|
| 0.01 s | 0.57 cm | 0.18 / 0.13 cm | 0.22 / 0.17 cm |
| 0.1 s | 4.91 cm | 1.95 / 1.70 cm | 2.05 / 1.76 cm |
| 0.5 s | 23.5 cm | 14.6 / 13.7 cm | 13.9 / 13.1 cm |
| 1 s | 43.4 cm | 36.8 / 35.0 cm | 34.6 / 33.6 cm |

- **Both are sensible from the 2nd sample (0.02 s after the start)** at every horizon (error within 15 % of the
  steady state).
- After one sample neither filter knows the velocity, so both predict "the mosquito stays" (identical errors).
- The measurements are so precise that two of them already give a good velocity; one sample ahead the
  error after two samples is even below the steady state, because a fresh 2-point velocity is the least smoothed.
- Consequence for the evaluation protocol: the 1 s of history before the first scored prediction is more than
  enough for both filters.
