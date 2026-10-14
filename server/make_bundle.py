"""Pack everything the server needs for the LSTM vs GRU study into lstm_server_bundle.zip (repository root).

The data (*.npz, *.csv) are not in git, so a git clone on the server is not enough; the bundle has the code,
Data/ML_dataset, Data/Benchmark, the selected GRU (= GRU baseline seed 0) and the results already computed here
(constant velocity, GRU baseline seed 0 evaluation, laptop timing).

Usage (from the repo root, on this computer):
    .venv/Scripts/python -m server.make_bundle
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lstm_server_bundle.zip"


def files():
    for p in sorted((ROOT / "Algorytms").rglob("*")):
        if p.suffix in (".py", ".md") and "__pycache__" not in p.parts:
            yield p
    for d in ("Data/ML_dataset", "Data/Benchmark", "server"):
        yield from (p for p in sorted((ROOT / d).iterdir()) if p.is_file() and p.suffix != ".pyc")
    for f in ("requirements.txt", "Results/gru/disp_h32_k10/model.pt", "Results/gru/disp_h32_k10/history.json",
              "Results/lstm/timing_laptop.json", "Results/lstm/constant_velocity/eval.json",
              "Results/lstm/gru_base/seed0/model.pt", "Results/lstm/gru_base/seed0/history.json",
              "Results/lstm/gru_base/seed0/eval.json"):
        if (ROOT / f).exists():
            yield ROOT / f


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            name = p.relative_to(ROOT).as_posix()
            if p.suffix == ".sh":      # Linux line endings, whatever git did on Windows
                z.writestr(name, p.read_bytes().replace(b"\r\n", b"\n"))
            else:
                z.write(p, name)
        n = len(z.namelist())
    print(f"{OUT.name}: {n} files, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
