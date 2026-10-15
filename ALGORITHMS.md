# Algorithms tested in this project

This file lists every prediction algorithm tested on the Aedes aegypti 3D flight data: what each one is, which
variants were tried, how they performed, and whether they were kept. Details, code and full results are in the
README of each algorithm (linked in every section).

Status used below:

- ✅ **selected:** the variant that represents its algorithm in the final benchmark;
- 🔬 **experiment:** tested and documented, but not selected;
- ❌ **rejected:** clearly worse or unusable.

## Common test protocol

All algorithms are scored the same way ([Algorytms/README.md](Algorytms/README.md)):

| item | value |
|---|---|
| test set | the 20 longest trajectories in `Data/Benchmark/` (17.2 min of flight, 100 frames per second) |
| training data | anything learned or estimated comes only from the other 2 365 trajectories |
| prediction points | every measured frame after 1 s of history, ~72 000 per horizon |
| horizons | the model predicts 1 … 500 frames ahead, i.e. **10 ms … 5 s** |
| error | Euclidean distance in cm to the true position; interpolated frames are never used as ground truth |

## Summary

Mean position error [cm] on the benchmark:

| algorithm | type | variants tested | selected variant | 0.01 s | 0.1 s | 0.5 s | 1 s | 5 s | status |
|---|---|---|---|---|---|---|---|---|---|
| Constant velocity | baseline | 1 | last displacement repeated | 0.15 | 1.83 | 13.65 | 34.83 | 256 | reference |
| Linear autoregression | baseline, learned | 1 | 20 displacements → next one | 0.14 | 2.08 | 37.87 | 258.7 | diverges | ❌ |
| Kalman filter | classical filter | 2 models + 10 settings | constant velocity, 0.336 m/s | 0.17 | 1.76 | 13.09 | 33.60 | 248 | ✅ |
| Particle filter | classical filter | 60 noise pairs + 36 size configs | N = 5000, keep 100 %, 0.504 m/s, likelihood 0.2 cm | 0.22 | 2.05 | 13.87 | 34.62 | 249 | ✅ |
| GRU | neural network | 4 inputs + 6 experiments (10 networks) | velocity input, hidden 32, 10-step output | 0.11 | 0.98 | 10.87 | 32.56 | 343 | ✅ |
| LSTM | neural network | 7 variants × 5 seeds (35 networks) | LSTM B: velocity + acceleration input, hidden 32, 10-step output (seed 4) | 0.11 | **0.89** | **10.15** | 27.83 | 190 | ✅ |
| Transformer | neural network | 3 variants × 5 seeds (15 networks) | T3: Pre-LN encoder, d 20, window 1 s in 10 patches, 10-step output (seed 3) | 0.12 | 1.15 | 10.53 | **27.61** | **174** | ✅ |

**Final benchmark** ([Results/benchmark/](Results/benchmark/README.md)): Kalman filter, particle filter, linear
autoregression, GRU, LSTM and transformer, side by side.

- **10 ms – 0.86 s:** the LSTM is the best model.
- **0.87 s – 5 s:** the transformer is the best model.

All of them, plus constant velocity, are also in the full comparison
([Results/comparison/](Results/comparison/comparison_table.md)) and the interactive 3D viewer
(`Results/viewer/index.html`).

---

## 1. Constant velocity (baseline)

[Algorytms/baselines/](Algorytms/baselines/README.md) · results `Results/baselines/constant_velocity/`

- **Idea:** the mosquito keeps the velocity it had over the last frame:
  p(t + h) = p(t) + h · (p(t) − p(t − 1)). No parameters, no training.
- **Tested:** 1 variant.
- **Result:**
  - One frame ahead it beats the Kalman filter (0.15 vs 0.17 cm), because the tracker's positions are already
    smooth.
  - From 0.1 s on it is only 1–4 % behind the Kalman filter.
  - Above ~1.5 s it beats most of the neural networks.
- **Role:** the minimum every algorithm has to beat.

## 2. Linear autoregression (baseline) ❌

[Algorytms/baselines/](Algorytms/baselines/README.md) · results `Results/baselines/linear_ar/`

- **Idea:** the next displacement is a linear function of the last 20 displacements of all three axes (60 inputs
  + bias → 3 outputs).
  - Fitted by least squares on the training split (828 426 samples).
  - Multi-step forecasts feed each prediction back into the input window.
- **Tested:** 1 variant.
- **Result:**
  - It is the best one-step predictor of the classical models (0.14 cm), but **unstable when iterated**.
  - The fitted model has spectral radius 1.03, so its error grows like 1.03ⁿ: 2.6 m at 1 s, 8·10⁵ m at 5 s.
- **Conclusion:** a model fitted one step ahead is not automatically a good multi-step model. The same problem
  appears, in a milder form, in the one-step GRU.

## 3. Kalman filter ✅

[Algorytms/kalman/](Algorytms/kalman/README.md) · results `Results/kalman/`

- **Idea:**
  - State [x, y, z, vx, vy, vz]; constant-velocity motion with white-noise acceleration.
  - One tuning value, the **velocity setting s**: initial velocity std = s, process noise q = s² / 1 s.
  - Measurement noise is estimated from the training data (0.0076 / 0.0072 / 0.0512 cm).
  - Interpolated frames are treated as measurements with 1 cm noise.
  - The forecast is the constant-velocity extrapolation of the filtered state.
- **Tested:**

  | test | variants | outcome |
  |---|---|---|
  | motion model, at s = 0.84 m/s (half the 99.9th-percentile speed) | constant velocity (CV) vs constant acceleration (CA) | CA is worse at every horizon (1.76 m vs 0.35 m at 1 s), because it extrapolates noise as acceleration. ❌ CA dropped. |
  | velocity setting | 10 values evenly spaced over (0, v_max]: 0.168 … 1.679 m/s | **0.336 m/s** chosen: best mean over 12 horizons. The setting matters little: the worst is 4–7 % behind the best. |
  | convergence after a cold start (961 starts) | – | sensible from the **2nd sample** (0.02 s) |

- **Result (0.336 m/s):**

  | horizon | 0.1 s | 0.5 s | 1 s | 5 s |
  |---|---|---|---|---|
  | mean error | 1.76 cm | 13.1 cm | 33.6 cm | 2.48 m |

  Up to ~0.1 s it is accurate: 98 % of predictions within 5 cm. Beyond ~0.5 s no constant-velocity model can follow
  the mosquito's turns.

## 4. Particle filter ✅

[Algorytms/particle/](Algorytms/particle/README.md) · results `Results/particle/`

- **Idea:**
  - Each particle is a state [x, y, z, vx, vy, vz]; the motion model is the same as the Kalman filter's.
  - Particles are weighted by a Gaussian likelihood of the measurement.
  - **Elitist resampling:** keep the best fraction of particles, then redraw all N from them.
  - **The forecast comes from the most probable particle.**
- **Tested:**

  | test | variants | outcome |
  |---|---|---|
  | noise sweep (N = 1000) | process noise 0.168 … 1.679 m/s (10) × likelihood width 0.05 … 2 cm (6) = 60 | **0.504 m/s, likelihood 0.2 cm.** The real measurement noise is too narrow: one particle takes all the weight, so the likelihood must be wider. 0.168 m/s loses the mosquito. |
  | particles × elite fraction | N = 100 … 5000 (6) × keep 1 … 100 % (6) = 36 | **N = 5000, keep 100 %.** More particles always help, but most of the gain is below 500. Keeping only the best particles never helps. |
  | convergence after a cold start | – | sensible from the **2nd sample**, like the Kalman filter |

- **Result:** 2.05 cm at 0.1 s, 34.6 cm at 1 s, 2.49 m at 5 s.
  - Close to but **slightly worse than the Kalman filter at every horizon**, which is expected: for this linear
    Gaussian model the Kalman filter is optimal.
  - About 40× slower than the Kalman filter.

## 5. GRU (recurrent neural network) ✅

[Algorytms/gru/](Algorytms/gru/README.md) · results `Results/gru/`

- **Idea:** a small GRU (1 layer) reads the last frames of motion and predicts the next motion.
  - Inputs are normalised by training-split statistics.
  - Longer forecasts: predictions are appended to the input window and the network is run again.
  - Training: Adam (lr 1e-3), batch 64, at least 20 epochs, early stopping on the validation split, CPU only.
- **Tested, part 1: four input representations** (hidden 32, window 20, one-step output):

  | network | input → output | 0.1 s | 1 s | status |
  |---|---|---|---|---|
  | velocity only | displacements → next displacement | 1.21 | 38.9 | 🔬 reference of the experiments |
  | position + velocity + acceleration | 9 features → next displacement | 1.27 | 55.9 | 🔬 best one-step (0.10 cm), iterates badly |
  | position only | positions → next position | 6.09 | 248 | ❌ |
  | position + velocity | 6 features → next position | 8.72 | 259 | ❌ (retrained with a plateau stopping rule; did not help) |

  Predicting the change (velocity) works; predicting the absolute position does not.

- **Tested, part 2: experiments on the velocity-only GRU**, one change at a time:

  | variant | parameters | 0.1 s | 0.5 s | 1 s | 5 s | status |
  |---|---|---|---|---|---|---|
  | hidden 16 | 1 059 | 1.40 | 14.5 | 40.7 | 283 | 🔬 small, almost as good |
  | hidden 64 | 13 443 | 1.08 | 15.1 | 47.2 | 540 | 🔬 better one step, iterates worse |
  | window 10 frames | 3 651 | 1.20 | 12.9 | 33.6 | 227 | 🔬 most stable one-step GRU |
  | window 50 frames | 3 651 | 1.11 | 14.8 | 47.2 | 541 | 🔬 |
  | window 100 frames | 3 651 | 1.08 | 12.8 | 36.8 | 283 | 🔬 |
  | **10 steps output at once** | 4 542 | **0.98** | **10.9** | **32.6** | 343 | ✅ selected |

- **Selected:** GRU with 10-step output (`Results/gru/disp_h32_k10/`).
  - It predicts the next 0.1 s in one call, so it is iterated 10× less often.
  - Better than the Kalman filter by 44 % at 0.1 s, 17 % at 0.5 s and 3 % at 1 s.
  - Worse beyond ~1 s: 38 % worse at 5 s.

## 6. LSTM (recurrent neural network) ✅

[Algorytms/lstm/](Algorytms/lstm/README.md) · results `Results/lstm/`

- **Idea:** an LSTM instead of the GRU, in a controlled study.
  - Every variant changes one thing of the selected GRU.
  - Every variant is trained **5 times with different seeds**.
  - Same data, normalisation, training, output head (10 steps) and evaluation as the GRU.
  - LSTM forget-gate bias = 1.
  - Trained on the server: 35 runs, ~8.5 h on 6 cores.
- **Decision rule:** a variant counts as better only when the difference of the means is larger than the spread
  between seeds.
- **Tested** (mean ± std over 5 seeds):

  | variant | change vs GRU | parameters | ADE 0.1 s | 1 s | 5 s | vs GRU baseline |
  |---|---|---|---|---|---|---|
  | GRU baseline | – (5 seeds of the selected GRU) | 4 542 | 0.531 ± 0.010 | 31.7 ± 1.0 | 317 ± 26 | reference |
  | LSTM A | GRU → LSTM, hidden 32 | 5 726 | 0.500 ± 0.007 | 30.7 ± 0.9 | 273 ± 37 | 🔬 better everywhere |
  | LSTM A' | GRU → LSTM, hidden 28 (same parameter count) | 4 566 | 0.511 ± 0.009 | 32.5 ± 1.3 | 328 ± 45 | 🔬 better only at 0.1 s |
  | **LSTM B** | LSTM + acceleration input (6 features) | 6 110 | **0.478 ± 0.004** | **29.1 ± 1.5** | **226 ± 61** | ✅ better everywhere |
  | GRU-B | GRU + acceleration input (control) | 4 830 | 0.504 ± 0.007 | 30.4 ± 2.2 | 295 ± 89 | 🔬 better only at 0.1 s |
  | LSTM C50 | LSTM, window 50 frames | 5 726 | 0.496 ± 0.007 | 32.2 ± 1.8 | 341 ± 43 | 🔬 no gain |
  | LSTM C100 | LSTM, window 100 frames | 5 726 | 0.501 ± 0.004 | 32.4 ± 1.5 | 337 ± 43 | 🔬 no gain |

  ADE 0.1 s is the mean error over the 10 frames of the first block.

- **Conclusions:**
  - **LSTM cell:** helps, partly because the network is larger. At the same parameter budget it helps only up to
    0.1 s.
  - **Acceleration input:** helps both the LSTM and the GRU.
  - **Longer window:** gives nothing at any horizon.
  - **Speed:** on the CPU the LSTM is ~3× faster per call than the GRU (0.25 vs 0.79 ms, laptop, batch 1).
- **Selected:** LSTM B, seed 4, chosen by validation loss (`Results/lstm/best/`).
  - **Best model of the whole project from 10 ms to 0.86 s:** 0.89 cm at 0.1 s, 10.2 cm at 0.5 s. Beyond that the
    transformer is better; the LSTM reaches 27.8 cm at 1 s and 1.90 m at 5 s.
  - Caveat: at 5 s the result depends strongly on the seed (226 ± 61 cm over the 5 seeds).

## 7. Transformer ✅

[Algorytms/transformer/](Algorytms/transformer/README.md) · results `Results/transformer/`

- **Idea:** a small Pre-LN transformer encoder.
  - Input: displacements, cut into patches of consecutive frames, one patch per token.
  - Learned positional encoding; the last token goes to a linear head that outputs 10 displacements.
  - Same data, normalisation, training, output and rollout as the GRU and LSTM.
  - 5 seeds per variant; trained on the server.
  - Literature: Giuliari et al. 2020 (plain transformer for trajectories), AeroCast 2026 (Pre-LN, last token,
    displacements), PatchTST 2023 (patching).
- **Tested** (mean ± std over 5 seeds):

  | variant | d / ff | window / patch | parameters | ADE 0.1 s | 1 s | 5 s | status |
  |---|---|---|---|---|---|---|---|
  | T1 | 20 / 40 | 20 frames / 1 | 4 570 | 0.660 ± 0.008 | 27.6 ± 0.2 | 160 ± 10 | 🔬 |
  | T2 | 32 / 64 | 20 frames / 1 | 10 366 | 0.661 ± 0.007 | 27.6 ± 0.9 | 169 ± 37 | 🔬 no gain from size |
  | **T3** | 20 / 40 | 100 frames (1 s) / 10 | 4 910 | **0.630 ± 0.006** | **27.3 ± 0.2** | 169 ± 6 | ✅ |
  | *GRU baseline / LSTM B* | | | | *0.531 / 0.478* | *31.7 / 29.1* | *317 / 226* | |

- **Conclusions:**
  - **Short horizons:** worse than the GRU and LSTM up to ~0.4 s. The transformers are under-fitted: the training
    loss stays at ~0.14, against ~0.11 for the RNNs.
  - **Long horizons:** better than the GRU from ~0.45 s and than LSTM B from ~0.8 s, with a very small seed spread.
  - **Size (T2 vs T1):** a larger network brings nothing.
  - **Patching (T3 vs T1):** the 1 s window in patches helps at 0.1–1 s, as in PatchTST.
- **Selected:** T3, seed 3, chosen by validation loss (`Results/transformer/best/`).
  - **Best model of the whole project from 0.87 s to 5 s:** 27.6 cm at 1 s, 1.74 m at 5 s.
  - It also has the smallest worst case at long horizons: 5.3 m maximum at 5 s.

## 8. Final benchmark

[Results/benchmark/](Results/benchmark/README.md)

The selected model of every algorithm side by side: tables of mean / median / p95 / max error and the share
within 5 / 10 cm, figures, and conclusions.

- **Short horizons:** the LSTM wins up to 0.86 s; at 0.1 s it is twice as accurate as the Kalman filter.
- **Long horizons:** the transformer wins from 0.87 s; at 5 s it is 30 % below the Kalman filter.
- **GRU:** falls behind the Kalman filter beyond ~1 s.
- **Particle filter:** slightly behind the Kalman filter everywhere.
- **Linear autoregression:** diverges.

---

## Number of evaluated configurations

| algorithm | configurations | what |
|---|---|---|
| Kalman filter | 12 | CV and CA models, plus 10 velocity settings |
| Particle filter | 97 | 60 noise pairs, 36 sizes, 1 full evaluation |
| Baselines | 2 | constant velocity, linear autoregression |
| GRU | 11 networks | 4 input representations, 6 experiments, 1 retrained position + velocity network |
| LSTM study | 35 networks | 7 variants × 5 seeds, one of them the selected GRU |
| Transformer study | 15 networks | 3 variants × 5 seeds |
| **total** | **~172** | |
