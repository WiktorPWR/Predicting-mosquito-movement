# LSTM vs GRU

> **Selected model: LSTM B, seed 4.** It enters the final benchmark next to the GRU, the Kalman filter and the
> particle filter.
>
> | property | value |
> |---|---|
> | evaluation output | [Results/lstm/best/](../../Results/lstm/best/summary.md) |
> | architecture | LSTM, hidden 32, window 20 |
> | input | displacements + accelerations (6 per frame) |
> | output | 10 displacements at once |
> | parameters | 6 110 |
> | how the seed was chosen | lowest validation loss of the 5 seeds, not on the test set |
>
> Mean error [cm]:
>
> | model | 0.1 s | 0.5 s | 1 s | 5 s |
> |---|---|---|---|---|
> | LSTM B, seed 4 | 0.89 | 10.15 | 27.8 | 190 |
> | GRU (10 steps) | 0.98 | 10.87 | 32.6 | 343 |
> | Kalman | 1.76 | 13.09 | 33.6 | 248 |
>
> Re-create the evaluation with `python -m Algorytms.lstm.run --best`.

Does an LSTM predict the mosquito's flight better than the selected GRU
([../gru/README.md](../gru/README.md), `Results/gru/disp_h32_k10`)? Every variant changes exactly one thing of
the GRU baseline and is trained 5 times (seeds 0–4), so differences can be compared with the spread between seeds.
Results: [Results/lstm/results.md](../../Results/lstm/results.md).

## Baseline (unchanged)

GRU baseline = the selected GRU:

- **Input:** the last 20 frame-to-frame displacements (0.2 s window, 10 ms frames), shape (batch, 20, 3). Each
  is divided by its training std: x 0.349, y 0.250, z 0.434 cm.
- **Network:** GRU (1 layer, hidden 32) → last hidden state (32) → Linear 32 → 30 → reshape to (batch, 10, 3).
- **Output:** the next 10 displacements (0.1 s) at once. Positions = last known position + cumulative sum of the
  displacements.
- **Horizons over 0.1 s:** the 10 predicted positions are appended to the window, the 10 oldest are dropped,
  and the network is called again.
- 4 542 parameters.

## Variants

| variant | change | input | hidden | window | parameters | question |
|---|---|---|---|---|---|---|
| GRU baseline | – | 3 | 32 | 20 | 4 542 | reference |
| LSTM A | GRU → LSTM | 3 | 32 | 20 | 5 726 | is an LSTM with the same hidden size better? |
| LSTM A' | GRU → LSTM, smaller | 3 | 28 | 20 | 4 566 | is A's gain due to LSTM or to the larger network? |
| LSTM B | + acceleration | 6 | 32 | 20 | 6 110 | does the acceleration input help? |
| GRU-B | + acceleration (control) | 6 | 32 | 20 | 4 830 | is B's gain due to the input or the architecture? |
| LSTM C | longer window | 3 | 32 | 50 | 5 726 | does a 0.5 s history help? |
| LSTM C | longer window | 3 | 32 | 100 | 5 726 | does a 1 s history help? |

- **Acceleration (B, GRU-B):** a_t = d_t − d_(t−1), where d are the frame-to-frame displacements. It is divided by
  its per-axis std from the training split only: 0.0958 / 0.0705 / 0.1999 cm per frame². A 20-frame window
  therefore needs 21 displacements (22 positions). In the rollout the accelerations are recomputed from the
  predicted positions appended to the window.
- **LSTM biases:** all zero, except the forget gate, where bias_ih + bias_hh = 0.5 + 0.5 = 1. PyTorch gate
  order: input, forget, cell, output. The memory therefore starts as "keep".
- **Implementation:** [model.py](model.py). One `RNNNet` class with cell `gru`/`lstm`; the forward pass uses the
  last hidden state `h[-1]`.

## Common rules (all variants, GRU baseline included)

- **Same data:** the same split into whole flights as the GRU (`Data/ML_dataset`, 2 010 train / 355 val
  trajectories), the same test set (the 20 benchmark trajectories) and the same normalisation.
- **Identical training:** constants and loss are imported from [../gru/train.py](../gru/train.py).
  - Adam, lr 1e-3, batch 64.
  - MSE over the measured target frames only (interpolated target frames masked out).
  - At least 20 epochs, then stop after 5 epochs without improvement of the validation loss, maximum 40.
  - The weights of the best epoch are kept.
- **Same head and rollout** for every variant.
- **5 seeds (0–4)** set both the weight initialisation and the order of the training samples.
  - The new training code with seed 0 reproduces the old GRU exactly (identical epoch 1 losses), so
    `Results/gru/disp_h32_k10` is used as GRU baseline seed 0.
- **Memory:** training windows are gathered from a per-frame feature table batch by batch. The numbers are the
  same as with materialised windows, but window 100 then fits in memory.

## Evaluation

[run.py](run.py) evaluates every model and seed. All models are scored on the **same test windows**:

- every measured frame of the 20 benchmark trajectories that has at least 100 displacements of history, i.e. the
  longest window;
- horizons 10 ms … 5 s;
- measured targets only;
- errors in cm on positions, after undoing the normalisation.

| metric | meaning |
|---|---|
| ADE 0.1 s | mean error over the 10 frames of the first block |
| FDE 0.1 s | error at the 10th frame (0.1 s) |
| 0.5 s / 1 s / 5 s | error after the rollout |
| time | one network call, batch 1, one CPU thread (median of 2 000 calls), laptop and Raspberry Pi |
| param., weights | number of parameters, size of the saved weights (state_dict) |

- **Reference:** constant velocity, i.e. the last displacement repeated over the whole horizon.
- **Table cells:** mean ± std over the seeds.
- **Figures:** mean over the seeds with the min–max range.

**Decision rule:**
- A variant counts as better or worse than another only when the difference of the means is larger than the
  spread between seeds, taken as the larger of the two standard deviations.
- If an LSTM wins by less than that, the GRU stays, because it is smaller and faster.

## Results

Computed on the server: 35 runs, about 8.5 h on 6 cores. Full table: [Results/lstm/results.md](../../Results/lstm/results.md).

Mean ± std over 5 seeds, in cm:

| model | param. | ADE 0.1 s | FDE 0.1 s | 0.5 s | 1 s | 5 s | time laptop [ms] |
|---|---|---|---|---|---|---|---|
| constant velocity | 0 | 0.961 | 1.827 | 13.65 | 34.83 | 256 | – |
| GRU baseline | 4 542 | 0.531 ± 0.010 | 0.982 ± 0.015 | 10.68 ± 0.17 | 31.72 ± 0.97 | 317 ± 26 | 0.79 |
| LSTM A (H32) | 5 726 | 0.500 ± 0.007 | 0.925 ± 0.014 | 10.47 ± 0.16 | 30.67 ± 0.88 | 273 ± 37 | 0.25 |
| LSTM A' (H28) | 4 566 | 0.511 ± 0.009 | 0.945 ± 0.017 | 10.72 ± 0.19 | 32.47 ± 1.30 | 328 ± 45 | 0.26 |
| **LSTM B (input 6)** | 6 110 | **0.478 ± 0.004** | **0.882 ± 0.007** | **10.24 ± 0.17** | **29.08 ± 1.48** | **226 ± 61** | 0.25 |
| GRU-B (input 6) | 4 830 | 0.504 ± 0.007 | 0.928 ± 0.012 | 10.43 ± 0.30 | 30.35 ± 2.21 | 295 ± 89 | 0.79 |
| LSTM C (window 50) | 5 726 | 0.496 ± 0.007 | 0.917 ± 0.013 | 10.54 ± 0.24 | 32.17 ± 1.79 | 341 ± 43 | 0.30 |
| LSTM C (window 100) | 5 726 | 0.501 ± 0.004 | 0.926 ± 0.006 | 10.63 ± 0.20 | 32.35 ± 1.50 | 337 ± 43 | 0.37 |

**Conclusions (decision rule: difference > spread between seeds):**

- **LSTM A (same hidden size):** better than the GRU baseline on every metric.
- **LSTM A' (same parameter budget):** better than the GRU baseline only at 0.1 s; equal at 0.5–5 s, and worse
  than A everywhere. So part of A's gain comes from the larger network. The LSTM cell itself helps at short
  horizons.
- **Acceleration input:**
  - helps both networks at 0.1 s (B vs A, GRU-B vs GRU baseline);
  - helps the LSTM also at 0.5 s and 1 s.
- **Same 6 inputs, LSTM B vs GRU-B:** the LSTM is better at 0.1 s; at longer horizons the difference is within
  the spread.
- **Longer window (50 or 100 frames):** no gain at any horizon; at 1–5 s it is even worse than window 20.
- **Winner: LSTM B.** It beats the GRU baseline on all five metrics by more than the seed spread. On the CPU it
  is also 3× faster per call (PyTorch's LSTM kernel is better optimised than its GRU kernel), despite 35 % more
  parameters.
- **Caveat at 5 s:** the spread between seeds is large (226 ± 61 cm). The selected seed 4 (190 cm) is the luckiest
  one at that horizon, although it was chosen on validation loss only.
- **Training length:** most runs reached the 40-epoch limit, but the validation loss changed only in the 4th
  decimal over their last 5 epochs. The limit therefore cut off practically nothing.

## Usage (from the repository root)

**The whole study on the server:** see [server/README.md](../../server/README.md). There,
`bash server/run_lstm_study.sh` in tmux does everything (CPU only, nice 19, resumable) and packs the results.

Single steps:

```
.venv/Scripts/python -m Algorytms.lstm.run_all [--parallel 4] [--longest-first]   # train all variants x seeds (resumable)
.venv/Scripts/python -m Algorytms.lstm.run_all status       # progress
.venv/Scripts/python -m Algorytms.lstm.train lstm_b --seed 3   # one run
    # options: --threads N, and outside the study: --hidden H --layers L --batch B --max-epochs E --checkpoint PATH
.venv/Scripts/python -m Algorytms.lstm.run [--jobs 4]       # evaluate, table and figures (also on partial results)
.venv/Scripts/python -m Algorytms.lstm.run --timing         # + laptop call time (only with nothing else running)
.venv/Scripts/python -m Algorytms.lstm.bench_train --save   # training-step speed: study variants + larger LSTMs, 1 vs all threads
python -m Algorytms.lstm.bench_cpu --save Results/lstm/timing_raspberry.json   # on the Raspberry Pi
```

- **Logs:** every run logs each epoch (loss, validation error, time) to `train.log` in its folder.
- **Option folders:** a run with changed `--hidden/--layers/--batch/--max-epochs` goes to its own folder
  (e.g. `lstm_a_h64_l2`) and is not part of the study table.

Notes on the commands:
- `run_all` runs seed by seed through all variants, and the window-100 LSTM last. It can be stopped at any time:
  finished runs are skipped and interrupted ones continue from `checkpoint.pt`.
- `bench_cpu` needs only numpy and torch. It uses freshly initialised networks, because the weights do not change
  the time.

## Outputs (`Results/lstm/`)

| path | content |
|---|---|
| `<variant>/seed<k>/model.pt` | trained network (+ normalisation, training history) |
| `<variant>/seed<k>/history.json` | loss and validation error per epoch, best epoch |
| `<variant>/seed<k>/train.log` | training log (time-stamped line per epoch) |
| `bench_train_<host>.md` | training-step benchmark of a machine |
| `<variant>/seed<k>/eval.json` | mean error per horizon, ADE, FDE |
| `results.md` | the results table and the decision-rule comparisons |
| `error_vs_horizon.png` | mean error vs horizon, one curve per model (mean of seeds, min–max band) |
| `relative_to_gru.png` | the same divided by the GRU baseline |
| `timing_laptop.json`, `timing_raspberry.json` | call times |
