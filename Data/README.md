# Data

From the raw tracker export to the benchmark every algorithm is tested on.
All scripts are in [Processing/](Processing/) and run from the repository root with the project's
virtual environment, e.g. `.venv/Scripts/python Data/Processing/stitch_trajectories.py`.
CSV files are not stored in git (`.gitignore`); every folder below can be regenerated from `Raw_data/`.

## Pipeline

| step | script | output | result |
|---|---|---|---|
| 0. raw export | — | `Raw_data/10-27-23-100-Mosquitoes-Empty-Chamber.csv` | 1 755 391 rows, 8 687 track ids, 1 214.7 s |
| 1. split by id | `split_by_mosquito.py` | `Split_data/<file>/<id>.csv` | 8 687 files |
| 2. stitch fragments | `stitch_trajectories.py` | `Stitched_data/<file>/traj_XXXXX.csv` | 3 496 trajectories |
| 3. fill gaps | `interpolate_gaps.py` | `Interpolated_data/<file>/traj_XXXXX.csv` | every trajectory sampled every 10 ms |
| 4. filter | `filter_trajectories.py` | `Filtered_data/<file>/traj_XXXXX.csv` | 2 385 trajectories kept |
| 5. benchmark | `make_benchmark.py` | `Benchmark/traj_XXXXX.csv` | the 20 longest trajectories (test set) |
| 6. ML dataset | `make_ml_dataset.py` | `ML_dataset/{train,val,test}.npz`, `norm.json`, `splits.json` | train / validation / test split for learned models |

Checks: `validate_stitching.py`, `validate_interpolation.py`; visual check: `plot_trajectories.py`.

## 0. Raw data

Columns: `id, datetime, x_cm, y_cm, z_cm, size_mm2, speed_mps, acceleration_mps2, distance_cm`.
100 mosquitoes in an empty chamber, 3D positions at 100 Hz.

What the data turned out to be:

- **`id` is a track fragment, not a mosquito.** The tracker loses a mosquito and re-acquires it under a new
  id. There are 8 687 ids; fragments last from 0.04 s to 33 s (median 1.0 s).
- Only **16 fragments are active at a time** (median, max 62) although 100 mosquitoes are in the chamber;
  resting mosquitoes are not tracked.
- A new fragment typically appears **~0.2 s** after one ends, **10–20 cm** away: the tracker's re-acquisition delay.
- The camera clock is not exactly 100 Hz: the frame period is **10.000044 ms**, so over 20 min the
  timestamps drift half a frame off a 10 ms grid. Frame numbers are therefore derived from timestamp differences.
- The tracker drops single frames inside fragments in **~11 %** of steps (steps of 2–5 frames).
- Positions are very smooth (probably filtered by the tracker): the noise estimated from second differences is
  ~0.008 cm in x and y and ~0.05 cm in z (depth).

## 1. Split by id

Every `id` written to its own file, rows sorted by time. Row count checked: the files together have exactly the
rows of the raw file.

## 2. Stitching fragments into trajectories

**Method** (`stitch_trajectories.py`). For every end of a fragment A and start of a fragment B with
0 < gap ≤ 1 s:

1. velocity at A's end and B's start = least-squares slope over 10 samples (0.1 s);
2. predict A forward and B backward over the gap with constant velocity;
3. cost = mean of both prediction errors / (8 cm + 60 cm/s × gap); pairs with cost > 1 are rejected;
4. all links are chosen at once by a **global optimal one-to-one assignment** (sparse bipartite matching,
   leaving an end or start unlinked costs 0.5), so conflicts between nearby mosquitoes are resolved optimally,
   not greedily.

**Test** (`validate_stitching.py`). Long fragments (known to be one mosquito) were cut by an artificial gap and the
whole dataset was stitched again; 400–600 cuts per gap length:

| gap | correct | wrong link | not linked |
|---|---|---|---|
| 0.1 s | 100.0 % | 0.0 % | 0.0 % |
| 0.2 s | 99.8 % | 0.2 % | 0.0 % |
| 0.3 s | 99.5 % | 0.2 % | 0.3 % |
| 0.5 s | 98.3 % | 0.5 % | 1.2 % |
| 0.8 s | 96.3 % | 0.7 % | 3.0 % |

The search radius growth was chosen from 40 / 60 / 90 cm/s: 40 misses many long gaps, 90 makes 2–6 % wrong links;
60 cm/s is the compromise (a wrong link mixes two mosquitoes, which is worse than a missed link).
Artificial gaps are easier than real ones (the tracker loses mosquitoes when they cross or turn sharply), so the
real error rate is somewhat higher.

**Result.** 8 687 fragments → **3 496 trajectories** (5 191 links; real gaps bridged: median 0.23 s, jump
median 13 cm). Duration median 1.0 s → 3.5 s, 90th percentile 6.3 s → 15.6 s, longest 33 s → 106 s.
`_links.csv` lists every link with gap, jump and cost; stitched files keep the original rows plus
`trajectory_id` and `segment` (index of the fragment in the trajectory).

## 3. Filling gaps

**Method** (`interpolate_gaps.py`). Every missing frame is filled with a **cubic Hermite curve** that matches
position and velocity (slope over 0.1 s) at both ends of the gap, i.e. the same constant-velocity assumption as
the stitching but smooth across the gap. Filled rows have `interpolated = 1`, empty `id`/`segment`;
speed, acceleration and step distance are recomputed from the curve, size is interpolated linearly.

**Test** (`validate_interpolation.py`). Measured frames were removed from long fragments and filled again;
500 cuts per gap length, mean position error of the filled frames:

| gap | Hermite | straight line |
|---|---|---|
| 0.03 s | 0.21 cm | 0.20 cm |
| 0.05 s | 0.29 cm | 0.27 cm |
| 0.1 s | 0.44 cm | 0.42 cm |
| 0.2 s | 0.69 cm | 0.74 cm |
| 0.5 s | 1.51 cm | 2.11 cm |
| 1.0 s | 3.49 cm | 5.61 cm |

(The mosquito moves ~0.6 cm per frame.) Equal for short gaps, 30–40 % better for the stitching gaps.

**Result.** 20.6 % of all frames are filled, mostly single dropped frames:

| gap length | gaps | share of filled frames |
|---|---|---|
| 1 frame | 198 920 | 44 % |
| 2–5 frames | 34 139 | 16 % |
| 0.06–0.2 s | 2 056 | 8 % |
| 0.2–0.5 s | 1 948 | 13 % |
| 0.5–1 s | 1 185 | 19 % |

## 4. Filtering

A model must learn mosquito flight, not our gap-filling curve. Single dropped frames are filled within the
measurement noise, so the criterion counts mainly the longer fills (`filter_trajectories.py`). A trajectory is kept when

- frames in fills longer than 0.05 s ≤ **10 %** of its frames, and
- all filled frames ≤ **30 %**.

**Result.** **2 385 of 3 496** trajectories kept, 68 % of the frames, ~249 min of flight, 487 trajectories ≥ 10 s.
In the kept data long fills are 4.0 % of the frames (16.7 % filled in total). `_filter_report.csv` has the
statistics of every trajectory and whether it was kept.

## 5. Benchmark (test set)

The **20 longest** filtered trajectories (39–106 s, 17.2 min in total) are frozen in `Benchmark/`. Every algorithm
is evaluated on exactly these; training data (`Algorytms/common/data.py: train_set`) is the other 2 365 trajectories.

## 6. Dataset for learned models (GRU, linear autoregression, ...)

`make_ml_dataset.py` → `ML_dataset/`. Split by **whole trajectories**:

| split | trajectories | frames | flight time | median / max duration | use |
|---|---|---|---|---|---|
| train | 2 010 | 1 187 824 | 198 min | 3.3 s / 38 s | fitting |
| validation | 355 | 206 241 | 34 min | 3.2 s / 33 s | early stopping, hyper-parameters |
| test | 20 | 103 005 | 17 min | 44.7 s / 106 s | the benchmark (same as Kalman / particle filter), final score only |

- The validation set is 15 % of the non-benchmark trajectories, drawn **per duration decile** (seed 0), so it has
  the same mix of short and long flights as the training set.
- Early stopping uses the **validation** set, never the test set: choosing the stopping point on the test set
  would leak it into the model.
- Each `.npz` holds all trajectories of the split concatenated: `pos` (N, 3) cm, `disp` (N, 3) frame-to-frame
  displacement (0 on the first frame of each trajectory), `measured` (N,), `offsets` (trajectory boundaries), `ids`.
- `norm.json`: per-axis displacement std of the **training** split: 0.349 / 0.250 / 0.434 cm (x / y / z);
  the network input is `disp / std`. The mean displacement is ~0 (≤ 0.012 cm) and is not subtracted.

Windows are cut by `Algorytms/common/ml_data.py: make_windows(split, window, horizon)`: input = the last `window`
normalised displacements up to the origin frame, target = the next `horizon` displacements, plus a mask of which
target frames are real measurements (interpolated frames may be in the input, never learned as targets). The origin
frame must be measured and the window must not cross a trajectory boundary. With window 20:

| split | samples, horizon 1 | samples, horizon 10 (measured target steps) |
|---|---|---|
| train | 828 426 | 940 143 (85.7 %) |
| validation | 143 228 | 162 641 (85.7 %) |
| test | 73 755 | 84 347 (86.2 %) |

The final comparison with the other algorithms is still done with the common benchmark protocol
(`Algorytms/common/evaluation.py`, horizons 10 ms – 5 s); the test windows are only for quick loss checks.
