"""From which sample on does a model give sensible predictions?

Cold starts: in every test trajectory the model is started from scratch every
COLD_START_EVERY_S seconds (at a measured frame) and fed only the frames from
that point on. For k = 1 .. WINDOW samples after the start we record the
prediction error at a few horizons and average it over all cold starts.

"Sensible from sample k" = first k from which the mean error stays within
TOLERANCE of the steady-state error, i.e. the mean over the last STEADY samples
of the window. The raw curve is used (no smoothing): with ~1000 cold starts it is
smooth enough, and a centred smoothing would blur the first samples.
"""
import numpy as np
import pandas as pd

from Algorytms.common.data import FRAME_S

COLD_START_EVERY_S = 1.0
WINDOW = 300                 # samples after the start that are scored (3 s)
STEADY = 100                 # last samples of the window = steady state
HORIZONS = [1, 10, 50, 100]  # steps
TOLERANCE = 0.15


def cold_start_errors(model, trajs, window=WINDOW, horizons=HORIZONS, every_s=COLD_START_EVERY_S):
    """Mean error [cm] per (samples since start, horizon): DataFrame indexed by k = 1..window."""
    max_h = max(horizons)
    hs = np.array(horizons)
    sums = np.zeros((window, len(hs))); counts = np.zeros((window, len(hs)))
    n_starts = 0
    for tr in trajs:
        n = len(tr.pos)
        for s0 in np.arange(0, n - window - max_h, int(every_s / FRAME_S)):
            while s0 < n and not tr.measured[s0]:
                s0 += 1
            seg = slice(s0, s0 + window + max_h)
            pos, meas = tr.pos[seg], tr.measured[seg]
            if len(pos) < window + max_h:
                continue
            origins = np.arange(window)            # k-1 = frames seen after the start frame
            preds = model.forecast(pos, meas, origins, max_h)[:, hs - 1]
            idx = origins[:, None] + hs
            err = np.linalg.norm(preds - pos[idx], axis=2)
            ok = meas[idx] & meas[origins][:, None]
            sums += np.where(ok, err, 0); counts += ok
            n_starts += 1
    df = pd.DataFrame(sums / counts, columns=[f"h{h}" for h in horizons])
    df.insert(0, "k", np.arange(1, window + 1))
    df.attrs["n_starts"] = n_starts
    return df


def sensible_from(df, horizons=HORIZONS):
    """First sample k after which the smoothed error stays within TOLERANCE of the steady state."""
    out = {}
    for h in horizons:
        e = df[f"h{h}"].to_numpy()
        steady = np.nanmean(e[-STEADY:])
        bad = np.flatnonzero(e > (1 + TOLERANCE) * steady)
        out[h] = dict(steady_cm=steady, first_k=int(bad[-1] + 2) if len(bad) else 1,
                      error_at_k1_cm=float(e[0]))
    return out
