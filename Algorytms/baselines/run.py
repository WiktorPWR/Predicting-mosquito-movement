"""Evaluate the reference models on the benchmark.

Usage (from the repo root):
    .venv/Scripts/python -m Algorytms.baselines.run
Writes Results/baselines/constant_velocity/ and Results/baselines/linear_ar/.
"""
import json

from Algorytms.baselines.models import ConstantVelocity, LinearAR
from Algorytms.common import data
from Algorytms.common.data import RESULTS_DIR
from Algorytms.common.evaluation import evaluate


def main():
    test = data.test_set()
    ar = LinearAR().fit()
    for model in (ConstantVelocity(), ar):
        st = evaluate(model, test, out=RESULTS_DIR / "baselines" / model.name)
        print(model.label)
        print(st[st["step"].isin([1, 10, 50, 100, 500])][["horizon_s", "mean_cm", "median_cm", "max_cm"]]
              .round(2).to_string(index=False))
    (RESULTS_DIR / "baselines" / "linear_ar" / "coefficients.json").write_text(
        json.dumps({"W": ar.W.tolist(), "rows": "20 x (dx, dy, dz) oldest first, then bias"}), encoding="utf-8")


if __name__ == "__main__":
    main()
