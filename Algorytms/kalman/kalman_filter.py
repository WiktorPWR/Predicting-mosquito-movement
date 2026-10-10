"""Constant-velocity (CV) Kalman filter for 3D mosquito position prediction.

State [x, y, z, vx, vy, vz]; the axes are independent but filtered jointly.

Tuning is a single velocity setting s (cm/s):
  * initial velocity std = s (direction unknown at start)
  * process noise: white-noise acceleration with q = s^2 / TAU_S, i.e. the
    velocity may drift by ~s within TAU_S seconds
  * measurement noise R from the data (second differences, per axis).

Interpolated frames are used as measurements with a larger noise INTERP_STD_CM
(the gap-filling error measured in Data/Processing/validate_interpolation.py),
so the filter never runs blind through a gap.
"""
import numpy as np

from Algorytms.common.data import FRAME_S

TAU_S = 1.0
INTERP_STD_CM = 1.0
SETTING_CM_S = 33.6   # chosen by the sweep (Results/kalman/sweep): 0.336 m/s


class KalmanCV:
    def __init__(self, meas_std, setting=SETTING_CM_S, dt=FRAME_S, tau=TAU_S, interp_std=INTERP_STD_CM, name="kalman_cv"):
        self.name = name
        self.label = f"Kalman CV ({setting / 100:.3f} m/s)"
        I3, Z3 = np.eye(3), np.zeros((3, 3))
        self.F = np.block([[I3, dt * I3], [Z3, I3]])
        self.H = np.hstack([I3, Z3])
        self.R = np.diag(np.asarray(meas_std, float) ** 2)
        self.R_interp = I3 * interp_std ** 2
        q = setting ** 2 / tau
        self.Q = q * np.block([[dt ** 3 / 3 * I3, dt ** 2 / 2 * I3], [dt ** 2 / 2 * I3, dt * I3]])
        self.P0 = np.diag(np.r_[np.asarray(meas_std, float) ** 2, np.full(3, setting ** 2)])
        self.params = dict(setting_cm_s=setting, tau_s=tau, q=q, meas_std_cm=list(np.round(meas_std, 5)),
                           interp_std_cm=interp_std)

    def filter(self, pos, measured):
        """Posterior state after each frame, shape (n, 6)."""
        n = len(pos)
        X = np.zeros((n, 6))
        x = np.r_[pos[0], np.zeros(3)]
        P = self.P0.copy()
        F, Q, H, I = self.F, self.Q, self.H, np.eye(6)
        X[0] = x
        for i in range(1, n):
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + (self.R if measured[i] else self.R_interp)
            K = np.linalg.solve(S, H @ P).T   # P H^T S^-1 (S symmetric)
            x = x + K @ (pos[i] - H @ x)
            P = (I - K @ H) @ P
            X[i] = x
        return X

    def forecast(self, pos, measured, origins, max_h):
        """Predicted positions for steps 1..max_h after each origin: (len(origins), max_h, 3).

        Constant velocity: p(t + h dt) = p(t) + v(t) h dt.
        """
        X = self.filter(pos, measured)[origins]
        h = np.arange(1, max_h + 1)[None, :, None] * FRAME_S
        return (X[:, None, :3] + X[:, None, 3:] * h).astype(np.float32)
