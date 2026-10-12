# Reference models (baselines)

Simple predictors on the same data and benchmark, the reference for the learned models (GRU).
Results: [Results/baselines/](../../Results/baselines/).

| file | purpose |
|---|---|
| `models.py` | `ConstantVelocity`, `LinearAR` |
| `run.py` | fits the linear autoregression on the training split and evaluates both → `Results/baselines/` (~1 min) |

## Models

- **Constant velocity:** the last displacement is repeated, p(t + h) = p(t) + h · (p(t) − p(t−1)). No parameters.
- **Linear autoregression (AR):** the next displacement is a linear function of the last 20 displacements of all
  three axes (60 inputs + bias → 3 outputs), fitted by least squares on the training split
  (828 426 one-step samples with a measured target, the same windows as the GRU). Multi-step forecasts feed the
  prediction back into the window, like the GRU.

## Results (benchmark, mean error)

| horizon | constant velocity | linear AR | *Kalman CV* |
|---|---|---|---|
| 0.01 s | 0.15 cm | **0.14 cm** | *0.17 cm* |
| 0.05 s | **0.85 cm** | 0.86 cm | *0.84 cm* |
| 0.1 s | 1.83 cm | 2.08 cm | ***1.76 cm*** |
| 0.2 s | 4.16 cm | 6.04 cm | ***3.98 cm*** |
| 0.5 s | 13.7 cm | 37.9 cm | ***13.1 cm*** |
| 1 s | 34.8 cm | 2.59 m | ***33.6 cm*** |
| 5 s | 2.56 m | diverges (8·10⁵ m) | ***2.48 m*** |

Full tables: `summary.md` in each result folder.

- **Constant velocity is a strong baseline:** one sample ahead it beats the Kalman filter (0.15 vs 0.17 cm,
  the tracker's positions are already smooth), and from 0.1 s on it is only 1–4 % behind it.
- **The linear AR is the best one-step predictor (0.14 cm) but unstable when iterated.** The fitted model's companion
  matrix has spectral radius 1.03 (3 eigenvalues outside the unit circle) and its coefficients on each axis sum to
  1.06–1.08: it slightly extrapolates accelerations. Over 1–2 steps that helps, over 500 steps the error grows like
  1.03⁵⁰⁰ ≈ 2·10⁶. A model fitted for one step is not automatically a good multi-step model — the same question
  applies to the GRU, which is also trained one step ahead and iterated.
