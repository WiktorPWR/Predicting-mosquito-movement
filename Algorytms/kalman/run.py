"""Evaluate the Kalman CV filter with the chosen setting (kalman_filter.SETTING_CM_S).

Results go to Results/kalman/cv_<setting>mps/.

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.kalman.run
"""
from Algorytms.common import data
from Algorytms.common.data import RESULTS_DIR
from Algorytms.common.evaluation import evaluate
from Algorytms.kalman.kalman_filter import KalmanCV


def main():
    meas_std = data.measurement_noise(data.train_set())
    model = KalmanCV(meas_std)
    out = RESULTS_DIR / "kalman" / f"cv_{model.params['setting_cm_s'] / 100:.3f}mps"
    st = evaluate(model, data.test_set(), out=out)
    print(model.label)
    print(st[st["step"].isin([1, 10, 50, 100, 200, 500])][["horizon_s", "mean_cm", "median_cm", "p95_cm", "max_cm"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
