"""Particle filter for 3D mosquito position prediction.

Each particle is a state [x, y, z, vx, vy, vz]. Per frame:
  1. predict: constant velocity + white-noise acceleration, independent per axis
     (same continuous model as the Kalman CV filter, q = setting^2 / TAU_S)
  2. weight:  Gaussian likelihood of the measured position (per-axis noise from
     the data; interpolated frames use INTERP_STD_CM like the Kalman filter)
  3. select:  the particle with the largest weight is the "best" particle; the
     forecast is its constant-velocity extrapolation
  4. resample (elitist): keep the `keep` fraction of particles with the largest
     weights and draw all N new particles from them (systematic resampling,
     proportional to weight). keep = 1.0 is the classic bootstrap filter.

The filter starts from the first frame with positions at the measurement and
velocities ~ N(0, setting^2) per axis, as the Kalman filter's P0.
"""
import numpy as np

from Algorytms.common.data import FRAME_S

TAU_S = 1.0
INTERP_STD_CM = 1.0


class ParticleFilter:
    def __init__(self, meas_std, setting, n_particles=1000, keep=1.0, seed=0, dt=FRAME_S, tau=TAU_S,
                 interp_std=INTERP_STD_CM, name="particle"):
        self.name = name
        self.n, self.keep, self.seed, self.dt = int(n_particles), float(keep), seed, dt
        self.label = f"Particle filter (N={self.n}, keep {self.keep:.0%}, {setting / 100:.3f} m/s)"
        self.meas_std = np.asarray(meas_std, float)
        self.interp_std = float(interp_std)
        self.setting = float(setting)
        self.q = setting ** 2 / tau
        # Cholesky factor of the per-axis CV process noise [[dt^3/3, dt^2/2], [dt^2/2, dt]] * q
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.L = np.linalg.cholesky(Q)
        self.params = dict(n_particles=self.n, keep=self.keep, setting_cm_s=self.setting, tau_s=tau, q=self.q,
                           meas_std_cm=list(np.round(self.meas_std, 5)), interp_std_cm=self.interp_std, seed=seed)

    def run(self, pos, measured, snapshot_at=(), n_snapshot=100):
        """Filter the whole sequence.

        Returns the best particle's state after each frame (n, 6) and, for every
        frame in `snapshot_at`, a subsample of the posterior cloud (best particle first)
        with normalised weights: {frame: (states (k, 6), weights (k,))}.
        """
        rng = np.random.default_rng(self.seed)
        rng_snap = np.random.default_rng(self.seed + 1)  # separate stream: snapshots must not change the run
        n, N = len(pos), self.n
        snap = set(int(s) for s in snapshot_at)
        P = np.empty((N, 6))
        P[:, :3] = pos[0]
        P[:, 3:] = rng.normal(0.0, self.setting, (N, 3))
        best = np.empty((n, 6))
        best[0] = np.r_[pos[0], 0, 0, 0]
        clouds = {}
        n_keep = max(1, int(round(self.keep * N)))
        L00, L10, L11 = self.L[0, 0], self.L[1, 0], self.L[1, 1]
        inv_var_meas = 1.0 / self.meas_std ** 2
        inv_var_interp = np.full(3, 1.0 / self.interp_std ** 2)
        u0 = (np.arange(N) + 0.0) / N
        for i in range(1, n):
            # 1. predict
            z = rng.standard_normal((N, 6))
            P[:, :3] += P[:, 3:] * self.dt + L00 * z[:, :3]
            P[:, 3:] += L10 * z[:, :3] + L11 * z[:, 3:]
            # 2. weight (log-likelihood, normalised in a numerically stable way)
            iv = inv_var_meas if measured[i] else inv_var_interp
            logw = -0.5 * (((P[:, :3] - pos[i]) ** 2) * iv).sum(1)
            w = np.exp(logw - logw.max())
            # 3. best particle
            b = int(np.argmax(w))
            best[i] = P[b]
            if i in snap:
                others = rng_snap.choice(np.delete(np.arange(N), b), size=min(n_snapshot - 1, N - 1), replace=False)
                idx = np.r_[b, others]
                clouds[i] = (P[idx].copy(), w[idx] / w.sum())
            # 4. elitist systematic resampling
            if n_keep < N:
                elite = np.argpartition(w, N - n_keep)[N - n_keep:]
            else:
                elite = np.arange(N)
            we = w[elite]
            cdf = np.cumsum(we)
            cdf /= cdf[-1]
            pick = np.searchsorted(cdf, u0 + rng.random() / N)
            P = P[elite[np.minimum(pick, n_keep - 1)]]
        return best, clouds

    def forecast(self, pos, measured, origins, max_h):
        """Constant-velocity extrapolation of the best particle: (len(origins), max_h, 3)."""
        best, _ = self.run(pos, measured)
        X = best[origins]
        h = np.arange(1, max_h + 1)[None, :, None] * FRAME_S
        return (X[:, None, :3] + X[:, None, 3:] * h).astype(np.float32)

    def cloud(self, pos, measured, frames, n_snapshot=100):
        """Posterior cloud subsamples at `frames` (same random stream as forecast, so the same run)."""
        return self.run(pos, measured, snapshot_at=frames, n_snapshot=n_snapshot)[1]
