"""Static figures: error vs prediction horizon, error over flight time, model comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Algorytms.common.data import FRAME_S

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(AXIS)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _fig(*args, **kw):
    return plt.subplots(*args, facecolor=SURFACE, **kw)


def plot_error_vs_horizon(st, title, path):
    fig, ax = _fig(figsize=(10, 5.5))
    h, c = st["horizon_s"], SERIES[0]
    ax.fill_between(h, st["p25_cm"], st["p75_cm"], color=c, alpha=0.18, lw=0, label="25–75th percentile")
    ax.plot(h, st["mean_cm"], color=c, lw=2, label="mean")
    ax.plot(h, st["median_cm"], color=c, lw=1.2, ls="--", label="median")
    ax.plot(h, st["p95_cm"], color=INK2, lw=1, label="95th percentile")
    ax.plot(h, st["max_cm"], color=MUTED, lw=1, ls=":", label="maximum")
    ax.set_yscale("log")
    ax.set_xlim(0, h.max())
    ax.set_xlabel("prediction horizon [s]", color=INK2)
    ax.set_ylabel("position error [cm] (log scale)", color=INK2)
    ax.set_title(title, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def plot_error_over_time(traj, origins, err, steps, title, path):
    """Top: error vs flight time for several horizons. Bottom: speed, to see what the mosquito did.

    Both are shown as a centred 0.2 s rolling mean; raw values are in the interactive viewer.
    """
    fig, (ax, axs) = _fig(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    t = origins * FRAME_S
    smooth = lambda v: pd.Series(v).rolling(20, center=True, min_periods=5).mean().to_numpy()
    for k, s in enumerate(steps):
        ax.plot(t, smooth(err[:, k]), color=SERIES[k], lw=1.5, label=f"horizon {s * FRAME_S:g} s")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 4)  # headroom for the legend
    ax.set_ylabel("position error [cm] (log)", color=INK2)
    ax.set_title(title + " (0.2 s rolling mean)", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, ncol=len(steps), loc="upper left", fontsize=9, bbox_to_anchor=(0, 1.0))
    _style(ax)
    axs.plot(traj.t, smooth(traj.speed), color=INK2, lw=1)
    axs.set_ylabel("speed [m/s]", color=INK2)
    axs.set_xlabel("flight time (moment of prediction) [s]", color=INK2)
    _style(axs)
    fig.tight_layout(); fig.savefig(path, dpi=110, facecolor=SURFACE); plt.close(fig)


def plot_model_comparison(stats_by_model, path, col="mean_cm", label="mean error", y_max_cm=1000):
    """Error vs horizon, one line per model; the y axis is clipped at y_max_cm so a diverging model
    does not squash the others."""
    fig, ax = _fig(figsize=(11, 5.5))
    clipped = False
    for k, (name, st) in enumerate(stats_by_model.items()):
        ax.plot(st["horizon_s"], st[col], color=SERIES[k], lw=2, label=name)
        clipped |= bool(st[col].max() > y_max_cm)
    ax.set_yscale("log")
    if clipped:
        ax.set_ylim(top=y_max_cm)
    ax.set_xlim(0, max(st["horizon_s"].max() for st in stats_by_model.values()))
    ax.set_xlabel("prediction horizon [s]", color=INK2)
    ax.set_ylabel(f"{label} [cm] (log scale)", color=INK2)
    ax.set_title(f"Model comparison: {label} vs prediction horizon"
                 + (f" (axis clipped at {y_max_cm / 100:g} m)" if clipped else ""), color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def plot_model_ratio(stats_by_model, reference, path, col="mean_cm", lo=0.5, hi=3.0):
    """Each model's error divided by the reference model's error at the same horizon (log y, clipped)."""
    ref = stats_by_model[reference][col].to_numpy()
    fig, ax = _fig(figsize=(11, 5.5))
    for k, (name, st) in enumerate(stats_by_model.items()):
        ax.plot(st["horizon_s"], np.clip(st[col].to_numpy() / ref, lo, hi), color=SERIES[k],
                lw=3 if name == reference else 2, label=name)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3) if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xscale("log")
    xt = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    ax.set_xticks(xt, [f"{t:g}" for t in xt])
    ax.set_xlabel("prediction horizon [s] (log)", color=INK2)
    ax.set_ylabel(f"error / {reference} error", color=INK2)
    ax.set_title(f"Model comparison relative to {reference} (below 1 = better; clipped at {hi:g})",
                 color=INK, loc="left", fontsize=11)
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


# ordinal blue ramp (light = small value), steps 250..700 of the reference sequential palette
ORDINAL_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def plot_sweep(df, out_dir, chosen=None):
    """Parameter sweep from a frame with `horizon_s` + one mean-error column per setting (ascending).

    Draws the absolute mean error and the mean error relative to the best setting per horizon;
    the `chosen` setting column is drawn thicker.
    """
    labels = [c for c in df.columns if c != "horizon_s"]
    ramp = [ORDINAL_BLUE[round(i * (len(ORDINAL_BLUE) - 1) / max(len(labels) - 1, 1))] for i in range(len(labels))]
    mean = df[labels].to_numpy().T
    h = df["horizon_s"].to_numpy()

    for kind, values, ylabel, fname in (
            ("abs", mean, "mean position error [cm] (log scale)", "sweep_mean_error.png"),
            ("rel", mean / mean.min(0), "mean error / best setting's mean error", "sweep_relative_error.png")):
        fig, ax = _fig(figsize=(10, 5.5))
        for lab, c, v in zip(labels, ramp, values):
            is_chosen = lab == chosen
            ax.plot(h, v, color=c, lw=3 if is_chosen else 1.5, zorder=3 if is_chosen else 2,
                    label=f"{lab} m/s" + ("  (chosen)" if is_chosen else ""))
        if kind == "abs":
            ax.set_yscale("log")
        else:
            ax.axhline(1, color=AXIS, lw=1, zorder=0)
        ax.set_xlim(0, h.max())
        ax.set_xlabel("prediction horizon [s]", color=INK2)
        ax.set_ylabel(ylabel, color=INK2)
        ax.set_title("Kalman CV: velocity setting sweep" + (" (relative to best per horizon)" if kind == "rel" else ""),
                     color=INK, loc="left")
        _style(ax)
        ax.legend(frameon=False, labelcolor=INK, title="setting", title_fontsize=9, fontsize=9,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=120, facecolor=SURFACE); plt.close(fig)


# sequential blue ramp, light = small value (steps 100..700 of the reference palette)
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def plot_heatmap(df, xcol, ycol, vcol, xlabel, ylabel, title, path, fmt="{:.2f}", clip=1.6):
    """Annotated heatmap of df[vcol] over the (xcol, ycol) grid; lower values are lighter. Best cell outlined.

    Colours saturate at `clip` x the best value (None: full range); the numbers in the cells are always exact.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    t = df.pivot_table(index=ycol, columns=xcol, values=vcol).sort_index(ascending=False)
    v = t.to_numpy()
    # colour scale clipped at clip x the best value so outliers do not wash out the good region
    vmax = max(np.nanmax(v), np.nanmin(v) * 1.0001)
    norm = LogNorm(vmin=np.nanmin(v), vmax=min(vmax, np.nanmin(v) * clip) if clip else vmax, clip=True)
    fig, ax = _fig(figsize=(1.1 * v.shape[1] + 3, 0.55 * v.shape[0] + 2))
    ax.imshow(v, cmap=LinearSegmentedColormap.from_list("seq", SEQ_BLUE), norm=norm, aspect="auto")
    bi, bj = np.unravel_index(np.nanargmin(v), v.shape)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            dark = norm(v[i, j]) > 0.55
            ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK, fontweight="bold" if (i, j) == (bi, bj) else "normal")
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=INK, lw=2))
    ax.set_xticks(range(v.shape[1]), [f"{c:g}" for c in t.columns])
    ax.set_yticks(range(v.shape[0]), [f"{r:g}" for r in t.index])
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def plot_convergence(curves, first_k, path, title):
    """Small multiples, one panel per horizon: mean error vs samples since a cold start, one line per model.

    Log x axis so the first samples (where convergence happens) are readable.
    curves: {model label: DataFrame from convergence.cold_start_errors}
    first_k: {model label: {horizon: {"first_k": k, ...}}} - marked with a vertical line
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    first = next(iter(curves.values()))
    hcols = [c for c in first.columns if c.startswith("h")]
    fig, axes = _fig(1, len(hcols), figsize=(4.2 * len(hcols), 4.4))
    for ax, hc in zip(np.atleast_1d(axes), hcols):
        h = int(hc[1:])
        for m, (label, df) in enumerate(curves.items()):
            ax.plot(df["k"], df[hc], color=SERIES[m], lw=1.6, label=label)
            ax.axvline(first_k[label][h]["first_k"], color=SERIES[m], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 300], ["1", "2", "5", "10", "20", "50", "100", "300"])
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"horizon {h * FRAME_S:g} s", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("samples since start (log)", color=INK2)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("mean position error [cm] (log)", color=INK2)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK, fontsize=9, loc="lower center",
               ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title + "  (dashed: sensible from this sample)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def plot_particles_effect(df, path, steps=(1, 10, 100)):
    """Mean error vs number of particles (log x), one line per elite fraction, one panel per horizon."""
    keeps = sorted(df["keep"].unique())
    ramp = [ORDINAL_BLUE[round(i * (len(ORDINAL_BLUE) - 1) / max(len(keeps) - 1, 1))] for i in range(len(keeps))]
    fig, axes = _fig(1, len(steps), figsize=(4.6 * len(steps), 4.6))
    for ax, s in zip(axes, steps):
        for k, c in zip(keeps, ramp):
            d = df[df["keep"] == k].sort_values("n_particles")
            ax.plot(d["n_particles"], d[f"h{s}"], color=c, lw=1.8, marker="o", ms=4, label=f"keep {k * 100:g} %")
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["n_particles"].unique()), [f"{n:g}" for n in sorted(df["n_particles"].unique())])
        ax.minorticks_off()
        ax.set_title(f"horizon {s * FRAME_S:g} s", color=INK, loc="left", fontsize=11)
        ax.set_xlabel("number of particles", color=INK2)
        _style(ax)
    axes[0].set_ylabel("mean position error [cm]", color=INK2)
    axes[-1].legend(frameon=False, labelcolor=INK, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Particle filter: effect of the number of particles and of the kept best fraction",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)


def plot_training_curve(history, best_epoch, title, path):
    """Left: train / validation loss per epoch (log). Right: validation one-step error in cm."""
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = _fig(1, 2, figsize=(11, 4.2))
    a1.plot(ep, [h["train_loss"] for h in history], color=SERIES[0], lw=1.8, marker="o", ms=3, label="train")
    a1.plot(ep, [h["val_loss"] for h in history], color=SERIES[1], lw=1.8, marker="o", ms=3, label="validation")
    a1.set_yscale("log"); a1.set_ylabel("MSE (normalised target, log)", color=INK2)
    a1.legend(frameon=False, labelcolor=INK)
    a2.plot(ep, [h["val_err_1step_cm"] for h in history], color=SERIES[1], lw=1.8, marker="o", ms=3)
    a2.set_ylabel("validation one-step error [cm]", color=INK2)
    for a in (a1, a2):
        a.axvline(best_epoch, color=MUTED, lw=1, ls="--")
        a.set_xlabel("epoch", color=INK2)
        _style(a)
    fig.suptitle(title + f"  (dashed: best epoch {best_epoch})", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=SURFACE); plt.close(fig)
