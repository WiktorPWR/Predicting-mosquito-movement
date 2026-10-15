# Transformer

> **Selected model: T3, seed 3.** It enters the final benchmark ([Results/benchmark/](../../Results/benchmark/README.md)).
>
> | property | value |
> |---|---|
> | evaluation output | [Results/transformer/best/](../../Results/transformer/best/summary.md) |
> | architecture | Pre-LN encoder, d 20, window 1 s in 10 patches of 10 frames |
> | output | 10 displacements at once |
> | parameters | 4 910 |
> | how the seed was chosen | lowest validation loss of the 5 seeds, not on the test set |
>
> Mean error [cm]:
>
> | model | 0.1 s | 0.5 s | 1 s | 5 s |
> |---|---|---|---|---|
> | Transformer T3, seed 3 | 1.15 | 10.53 | 27.6 | 174 |
> | LSTM B | 0.89 | 10.15 | 27.8 | 190 |
> | Kalman | 1.76 | 13.09 | 33.6 | 248 |
>
> It is the best model of the project from 0.87 s on. Re-create the evaluation with
> `python -m Algorytms.transformer.run --best`.

Do small transformer encoders predict the mosquito's flight better than the selected GRU and the best LSTM?
Three variants, each trained 5 times (seeds 0–4), compared with the GRU baseline and LSTM B from the
[LSTM vs GRU study](../lstm/README.md) on exactly the same test windows.
Results: [Results/transformer/results.md](../../Results/transformer/results.md).

## Literature basis

| source | finding | used here |
|---|---|---|
| Giuliari et al., *Transformer Networks for Trajectory Forecasting*, ICPR 2020, arXiv:2003.08111 | a plain transformer without interaction modelling was best on the TrajNet benchmark | a plain encoder is enough to start with |
| AeroCast, *Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN*, preprint 2026, arXiv:2606.25122 | input of 20 steps of 3D displacements, Pre-LN encoder, the last frame's token goes to the head, 5 s forecast | Pre-LN, last-token pooling, displacements instead of positions |
| Nie et al., *A Time Series is Worth 64 Words* (PatchTST), ICLR 2023, arXiv:2211.14730 | patching (several consecutive steps = one token) improves transformers and allows longer windows | variant T3: 1 s window in patches |

## Input, output, rollout (same as the GRU / LSTM)

- **Input:** the last `window` frame-to-frame displacements (10 ms frames), shape (batch, window, 3). Each is
  divided by its training std: x 0.349, y 0.250, z 0.434 cm.
- **Output:** the next 10 displacements (0.1 s) at once, shape (batch, 10, 3).
  Positions = last known position + cumulative sum of the displacements.
- **Horizons over 0.1 s:** the 10 predicted positions are appended to the window, the 10 oldest are dropped,
  and the network is called again.
- **Loss:** MSE over the measured target frames; interpolated target frames are masked out, as for the GRU and
  LSTM.

## Network ([model.py](model.py), `KomarTF`)

```
(batch, window, 3) → patching → (batch, window/patch, 3·patch)
  → Linear(3·patch → d) + learned positional encoding
  → Pre-LN TransformerEncoder (heads, feed-forward ff, dropout 0.1) → LayerNorm
  → last token → Linear(d → 30) → (batch, 10, 3)
```

- **Patching:** with patch > 1, consecutive frames are glued into one token. For example, (B, 100, 3) becomes
  (B, 10, 30): one token = 10 frames × 3 axes.
- **Constraints:** the number of heads must divide d, and the window must be divisible by the patch length.

| variant | d | ff | heads | layers | window | patch | tokens | parameters |
|---|---|---|---|---|---|---|---|---|
| T1: small transformer | 20 | 40 | 2 | 1 | 20 (0.2 s) | 1 | 20 | 4 570 |
| T2: larger transformer | 32 | 64 | 2 | 1 | 20 (0.2 s) | 1 | 20 | 10 366 |
| T3: longer window + patching | 20 | 40 | 2 | 1 | 100 (1 s) | 10 | 10 | 4 910 |

For comparison: GRU baseline 4 542 parameters, LSTM B 6 110.

## Training ([train.py](train.py))

Identical to the GRU / LSTM study; constants and loss are imported from there.

- Same data split.
- Adam, lr 1e-3, batch 64.
- At least 20 epochs, then stop after 5 epochs without improvement of the validation loss, maximum 40.
- The weights of the best epoch are kept.
- Dropout is active only in training.
- **5 seeds (0–4):** each sets the weight initialisation, the sample order and the dropout masks.
- **Checkpoint after every epoch:** includes the optimiser and random state, so an interrupted run resumes
  exactly where it stopped.
- **Instability fallback:** the literature advises a smaller lr or a warm-up if the loss jumps. `--lr` and
  `--warmup N` exist for that; runs that use them go to separate folders.

## Evaluation ([run.py](run.py))

Same test windows and metrics as the LSTM study:

- prediction points: every measured frame of the 20 benchmark trajectories with ≥ 100 displacements of history;
- ADE / FDE of the first 0.1 s block;
- rollout error at 0.5 / 1 / 5 s;
- call time (batch 1, one CPU thread), number of parameters, size of the weights.

The GRU baseline, LSTM B and constant velocity are read from `Results/lstm/` and not retrained.

**Decision rule** (as in the LSTM study): a variant counts as better or worse only when the difference of the
means is larger than the spread between seeds. Each transformer is compared against the GRU baseline and against
LSTM B. In addition, T2 vs T1 tests the network size, and T3 vs T1 tests the long window with patches.

## Results

Computed on the server: 15 runs, 6 in parallel. Seconds per epoch: T1 166, T2 212, T3 78.
Full table: [Results/transformer/results.md](../../Results/transformer/results.md).

Mean ± std over 5 seeds, in cm; GRU baseline and LSTM B from the LSTM study, same windows:

| model | param. | ADE 0.1 s | FDE 0.1 s | 0.5 s | 1 s | 5 s | time laptop [ms] | epochs (best) |
|---|---|---|---|---|---|---|---|---|
| constant velocity | 0 | 0.961 | 1.827 | 13.65 | 34.83 | 256 | – | – |
| GRU baseline | 4 542 | 0.531 ± 0.010 | 0.982 ± 0.015 | 10.68 ± 0.17 | 31.72 ± 0.97 | 317 ± 26 | 0.79 | 40 (39) |
| LSTM B | 6 110 | **0.478 ± 0.004** | **0.882 ± 0.007** | **10.24 ± 0.17** | 29.08 ± 1.48 | 226 ± 61 | 0.25 | 40 (38) |
| T1 (d 20) | 4 570 | 0.660 ± 0.008 | 1.203 ± 0.015 | 10.60 ± 0.05 | 27.58 ± 0.21 | **160 ± 10** | 0.31 | 24 (18) |
| T2 (d 32) | 10 366 | 0.661 ± 0.007 | 1.208 ± 0.013 | 10.55 ± 0.18 | 27.59 ± 0.86 | 169 ± 37 | 0.31 | 23 (18) |
| **T3 (window 1 s, patch 10)** | 4 910 | 0.630 ± 0.006 | 1.147 ± 0.012 | 10.46 ± 0.05 | **27.34 ± 0.19** | 169 ± 6 | 0.30 | 35 (32) |

**Conclusions (decision rule: difference > spread between seeds):**

- **Short horizons, up to ~0.4 s:** every transformer is worse than the GRU baseline and LSTM B.
  - At 0.1 s: ADE 0.63–0.66 vs 0.53 (GRU) and 0.48 (LSTM B).
  - The training loss settled close to the validation loss, at ~0.14 vs ~0.11 for the RNNs.
  - So the transformers are **under-fitted**, not over-fitted: they model the next frames less precisely.
  - T1 and T2 stopped after ~23 epochs (best epoch 18).
- **Long horizons:**
  - From ~0.45 s every transformer is better than the GRU, and from ~0.8 s also better than LSTM B.
  - At 1 s: 27.3–27.6 vs 29.1 cm (LSTM B). At 5 s: 160–169 vs 226 cm.
  - The smoother predictions do not build up errors when fed back block after block.
  - Seed spread at 5 s is small for T1 and T3 (± 6–10 cm, LSTM B ± 61), so this long-horizon result is reliable.
- **T2 vs T1:** no difference on any metric. 2.3× more parameters bring nothing; the model size is not the limit.
- **T3 vs T1:** the 1 s window in patches is better at 0.1 s, 0.5 s and 1 s, and equal at 5 s. As in PatchTST,
  patching makes a longer history usable. It is also the fastest variant to train: 10 tokens instead of 20.
- **Selected: T3.**
- **Speed:** all variants take ~0.3 ms per call on the laptop, like the LSTM and 2.6× faster than the GRU.
- **Possible next step (not done):** the under-fitting suggests training without dropout, or longer with a
  learning-rate schedule. This could improve the short horizons too.

![relative to GRU](../../Results/transformer/relative_to_gru.png)

## Usage

The whole study runs on the server, see [server/README.md](../../server/README.md):

```
.venv/Scripts/python -m server.make_bundle transformer          # on this computer → transformer_server_bundle.zip
tmux new -s tf 'bash server/run_transformer_study.sh 2>&1 | tee -a transformer_study.log'   # on the server
```

Single steps:

```
python -m Algorytms.transformer.run_all [--parallel 6]     # train 3 variants x 5 seeds (resumable)
python -m Algorytms.transformer.run_all status             # progress
python -m Algorytms.transformer.train tf_t3 --seed 2       # one run
python -m Algorytms.transformer.run [--jobs 6]             # evaluate, table and figures (also on partial results)
python -m Algorytms.transformer.bench_cpu --save Results/transformer/timing_raspberry.json   # on the Raspberry Pi
```

## Outputs (`Results/transformer/`)

| path | content |
|---|---|
| `<variant>/seed<k>/model.pt` | trained network (+ architecture, normalisation, training history) |
| `<variant>/seed<k>/history.json`, `train.log` | loss and validation error per epoch, best epoch |
| `<variant>/seed<k>/eval.json` | mean error per horizon, ADE, FDE |
| `results.md` | results table and the decision-rule comparisons |
| `error_vs_horizon.png`, `relative_to_gru.png` | mean of seeds with the min–max band, against the GRU baseline and LSTM B |
| `timing_laptop.json`, `timing_raspberry.json` | call times |
