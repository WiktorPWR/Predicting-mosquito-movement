# GRU (recurrent neural network)

> **Selected model: GRU with 10-step output (`Results/gru/disp_h32_k10/`)** — velocity input only, hidden 32,
> window 20, predicts the next 10 displacements at once. It is the GRU that enters the final benchmark of all
> algorithms. Load it with `Algorytms.gru.model.load("Results/gru/disp_h32_k10/model.pt", name="gru_k10")`.

Four small GRU networks that differ in **what they see and what they predict**, plus experiments with the
network size, input window and a 10-step output (see [Experiments](#experiments-network-size-input-window-multi-step-output)), trained on the same data and
compared with each other, the reference models ([../baselines/](../baselines/README.md)) and the Kalman / particle
filters on the benchmark. Results: [Results/gru/](../../Results/gru/); all models:
[Results/comparison/](../../Results/comparison/comparison_table.md).

| file | purpose |
|---|---|
| `model.py` | `GRUNet`, the four input representations (`Normaliser`), benchmark wrapper (`GRUPredictor`), save / load |
| `train.py` | training with early stopping → `Results/gru/<mode>_h<hidden>/model.pt`, `history.json`, `training_curve.png` |
| `run.py` | benchmark evaluation of the trained models |

```
.venv/Scripts/python -m Algorytms.gru.train disp      # velocity only            (~70 min on CPU, 40 epochs)
.venv/Scripts/python -m Algorytms.gru.train posonly   # position only
.venv/Scripts/python -m Algorytms.gru.train pos       # position + velocity
.venv/Scripts/python -m Algorytms.gru.train full      # position + velocity + acceleration
.venv/Scripts/python -m Algorytms.gru.run             # benchmark, ~8 min per model
```

## Data

`Data/ML_dataset/` (see [Data/README.md](../../Data/README.md#6-dataset-for-learned-models-gru-linear-autoregression-)):
split by whole trajectories into train (2 010 trajectories, 198 min), validation (355, 34 min) and test = the
20 benchmark trajectories. 100 frames per second. Window: the last 20 frames (0.2 s); ~828 000 training and
~143 000 validation samples; the target is always a measured frame (interpolated frames only appear in the input).
All normalisation statistics come from the training split, every input feature has std ≈ 1.

## Four input representations

| name (folder) | input per frame | features | output | positions of the forecast |
|---|---|---|---|---|
| **velocity only** (`disp_h32`) | velocity per axis (= displacement per 10 ms) / std; **no position** | 3 | next velocity | last position + sum of the predicted displacements |
| **position only** (`posonly_h32`) | position per axis, standardised; **no velocity** | 3 | next position | the network's output |
| **position + velocity** (`pos_h32`) | position (standardised) + velocity per axis | 6 | next position | the network's output |
| **position + velocity + acceleration** (`full_h32`) | position + velocity + acceleration (change of velocity between frames) per axis | 9 | next velocity | last position + sum of the predicted displacements |

Normalisation (training split): displacement std 0.349 / 0.250 / 0.434 cm (x / y / z), acceleration
(displacement change) std 0.096 / 0.071 / 0.200 cm per frame, positions standardised with the per-axis mean / std.

Network for all: `nn.GRU(input, hidden=32, num_layers=1, batch_first=True)` → `nn.Linear(32, 3)` on the last
hidden state (3 651 parameters with 3 features, 3 939 with 6, 4 227 with 9). Multi-step forecast: the predicted
frame is appended to the window, the oldest dropped and the network run again (sliding window, exactly the input it
was trained on); velocity and acceleration of the new frame are recomputed from the positions.

## Training

MSE on the normalised target, Adam (lr 1e-3), batch 64, one step ahead, CPU only (1 thread per run, runs in
parallel). The best epoch's weights (lowest validation loss) are always kept; the test set is never used.

Stopping rules (`STOPPING` in `train.py`), **every network trains at least 20 epochs**, at most 40:

- velocity only, position + velocity + acceleration: after 20 epochs, stop when the validation loss has not improved
  for 5 epochs;
- position only, position + velocity: their validation loss jumps from epoch to epoch, so after 20 epochs they stop
  when the **3-epoch moving average** of the validation loss has not improved by ≥ 1 % for 5 epochs.

| network | epochs | best epoch | validation one-step error |
|---|---|---|---|
| velocity only | 40 (limit) | 38 | 0.134 cm (epoch 1) → **0.114 cm** |
| position only | 28 | 28 | **0.241 cm** |
| position + velocity | 21 | 21 | **0.263 cm** (first run with 5-epoch patience: stopped at 12, best 0.248 cm) |
| position + velocity + acceleration | 40 (limit) | 40 | 0.128 cm (epoch 1) → **0.107 cm** |

- The networks that predict a velocity train smoothly without overfitting (validation loss ≤ training loss) and were
  still improving very slowly at the 40-epoch limit.
- The networks that predict a position: the training loss falls steadily, the validation error jumps between
  ~0.23 and 0.45 cm from epoch to epoch for the whole run. After standardisation one frame of motion (~0.5 cm) is
  ~1 % of the output scale, so tiny weight changes move the predicted position by millimetres. More epochs
  (retraining position + velocity with the 20-epoch minimum) did not help: one-step 0.248 → 0.263 cm on validation,
  benchmark 0.1 s error 5.35 → 8.72 cm.

## Results (benchmark, mean error)

| horizon | velocity only | pos + vel + acc | position only | pos + vel | *Kalman CV* | *constant velocity* |
|---|---|---|---|---|---|---|
| 0.01 s | 0.11 cm | **0.10 cm** | 0.23 cm | 0.25 cm | *0.17 cm* | *0.15 cm* |
| 0.05 s | 0.55 cm | **0.54 cm** | 2.08 cm | 2.50 cm | *0.84 cm* | *0.85 cm* |
| 0.1 s | **1.21 cm** | 1.27 cm | 6.09 cm | 8.72 cm | *1.76 cm* | *1.83 cm* |
| 0.2 s | **3.35 cm** | 3.80 cm | 20.8 cm | 52.3 cm | *3.98 cm* | *4.16 cm* |
| 0.5 s | 14.2 cm | 18.2 cm | 1.23 m | 1.80 m | ***13.1 cm*** | *13.7 cm* |
| 1 s | 38.9 cm | 55.9 cm | 2.48 m | 2.59 m | ***33.6 cm*** | *34.8 cm* |
| 2 s | 94.2 cm | 1.48 m | 3.09 m | 3.04 m | ***83.5 cm*** | *86.4 cm* |
| 5 s | 2.72 m | 3.89 m | 4.01 m | 3.12 m | ***2.48 m*** | *2.56 m* |

Maximum error of the velocity-only GRU: 1.0 cm one sample ahead, 12.3 cm at 0.1 s, 1.31 m at 1 s.
Full tables: `summary.md` in each result folder; figure `relative_to_kalman.png` in `Results/comparison/`.

## Conclusions

- **Predicting the change (velocity) works, predicting the absolute position does not.** The two networks that
  output a velocity are 5–7× better at 0.1 s than the two that output a position, whatever the position networks get
  as input (adding velocity to the position input did not help). Positions carry little information about the next
  move, and hitting an absolute position needs millimetre precision on a metre scale.
- **Velocity-only GRU: the best model of the whole comparison from ~0.1 to ~0.3 s** and within 0.01 cm of the best
  below that: 35 % better than the Kalman filter one sample ahead (0.11 vs 0.17 cm), 31 % better at 0.1 s
  (1.21 vs 1.76 cm). It has learned how the flight continues over a few frames, which a constant-velocity model
  cannot express.
- **Adding position and acceleration (9 features) gives the best one-step model (0.10 cm, best up to 0.05 s) but is
  worse from 0.1 s on**, and much worse at long horizons (56 vs 39 cm at 1 s). The extra inputs help to fit the next
  frame, but when the network is iterated on its own predictions the recomputed acceleration feeds errors back
  faster, and the position input lets it rely on where the mosquito is rather than on how it moves.
- **The reference GRU (hidden 32, window 20, one-step output) loses to the Kalman filter beyond ~0.35 s**
  (10–16 % worse from 0.5 s on). It is trained one step ahead and iterated up to 500 times, so small systematic
  errors accumulate (a milder version of the linear autoregression's divergence). The experiments below show
  which changes fix that.

## Experiments: network size, input window, multi-step output

Velocity-only GRU, **one change at a time** against the reference (hidden 32, window 20 frames, one-step output).
Same data, training rules (≥ 20, ≤ 40 epochs, early stopping on validation) and benchmark as above.
Command: `train disp --hidden H --window W --steps K`; folders `disp_h<H>[_w<W>][_k<K>]`.
Figure: [experiments_relative_to_kalman.png](../../Results/gru/experiments_relative_to_kalman.png).

Training was interrupted once (stopped at a fixed time) and resumed from per-epoch checkpoints
(`checkpoint.pt`: weights, optimiser, random state, history); a resumed run continues exactly where it stopped.

| variant | parameters | epochs (best) | val. 1-step | 0.01 s | 0.05 s | 0.1 s | 0.2 s | 0.5 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hidden 16 | 1 059 | 40 (40) | 0.123 cm | 0.11 | 0.64 | 1.40 | 3.58 | 14.5 | 40.7 | 100.3 | 283 |
| **reference** (hidden 32, window 20) | 3 651 | 40 (38) | 0.114 cm | 0.11 | 0.55 | 1.21 | 3.35 | 14.2 | 38.9 | 94.2 | 272 |
| hidden 64 | 13 443 | 40 (38) | 0.103 cm | **0.10** | 0.48 | 1.08 | 3.19 | 15.1 | 47.2 | 147.6 | 540 |
| window 10 (0.1 s) | 3 651 | 40 (40) | 0.118 cm | 0.11 | 0.57 | 1.20 | 3.18 | 12.9 | 33.6 | **79.3** | **227** |
| window 50 (0.5 s) | 3 651 | 40 (40) | 0.110 cm | **0.10** | 0.51 | 1.11 | 3.21 | 14.8 | 47.2 | 145.6 | 541 |
| window 100 (1 s) | 3 651 | 40 (39) | 0.111 cm | **0.10** | 0.51 | 1.08 | 3.00 | 12.8 | 36.8 | 94.8 | 283 |
| **10-step output** | 4 542 | 39 (34) | 0.118 cm | 0.11 | **0.47** | **0.98** | **2.62** | **10.9** | **32.6** | 94.5 | 343 |
| *Kalman CV* | | | | *0.17* | *0.84* | *1.76* | *3.98* | *13.1* | *33.6* | *83.5* | *248* |

Mean position error [cm] on the benchmark; best GRU per horizon in bold.

- **10 steps at once is the best model of the whole project up to ~1 s**: 44 % better than the Kalman filter at
  0.1 s (0.98 vs 1.76 cm), 17 % at 0.5 s (10.9 vs 13.1 cm) and 3 % at 1 s (32.6 vs 33.6 cm). Trained to predict
  0.1 s ahead directly, it is iterated 10× less often, so far fewer errors feed back. Beyond ~1 s it falls behind
  (each block of 10 is still fed back) and at 5 s it is 38 % worse than Kalman.
- **More capacity (hidden 64, window 50) fits the next frames better but iterates worse.** Both are the best one-step
  models (0.10 cm) and 8–11 % better than the reference at 0.1 s, but from ~0.5 s on their error grows much faster
  (47 cm at 1 s, 5.4 m at 5 s, twice the Kalman filter): a more flexible model reproduces its own small errors more
  confidently when fed its own predictions.
- **A short window (10 frames) is the most stable iterated model**: as good as the reference up to 0.1 s and the only
  one-step GRU that matches the Kalman filter from 0.5 s on (12.9 / 33.6 / 79.3 / 227 cm at 0.5 / 1 / 2 / 5 s; 8 %
  better than Kalman at 5 s). Less history to over-fit means smoother, more cautious extrapolation.
- **A long window (100 frames = 1 s) helps short horizons** (1.08 cm at 0.1 s, 12.8 cm at 0.5 s) without the
  long-horizon blow-up of window 50, but costs ~5× training time (~9 min per epoch) and inference.
- **Hidden 16 (1 059 parameters) is only 0–16 % worse than hidden 32** up to 1 s: a realistic candidate for the
  Raspberry Pi.
- Practical choice: **10-step output for horizons up to ~1 s**, the Kalman filter (or the window-10 GRU) beyond.
