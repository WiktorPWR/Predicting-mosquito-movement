"""Pack everything the server needs for one study into <study>_server_bundle.zip (repository root).

The data (*.npz, *.csv) are not in git, so a git clone on the server is not enough. Every bundle has the code,
requirements.txt, Data/ML_dataset, Data/Benchmark and server/; plus per study:
  lstm         the selected GRU (= GRU baseline seed 0) and the results already computed here
  transformer  the evaluations of the reference models from the LSTM study (constant velocity, GRU baseline and
               LSTM B, 5 seeds each) and the laptop timings, so the server's table compares against them

Usage (from the repo root, on this computer):
    .venv/Scripts/python -m server.make_bundle lstm
    .venv/Scripts/python -m server.make_bundle transformer
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY_FILES = {
    "lstm": ["Results/gru/disp_h32_k10/model.pt", "Results/gru/disp_h32_k10/history.json",
             "Results/lstm/timing_laptop.json", "Results/lstm/constant_velocity/eval.json",
             "Results/lstm/gru_base/seed0/model.pt", "Results/lstm/gru_base/seed0/history.json",
             "Results/lstm/gru_base/seed0/eval.json"],
    "transformer": ["Results/lstm/constant_velocity/eval.json", "Results/lstm/timing_laptop.json",
                    "Results/transformer/timing_laptop.json"]
                   + [f"Results/lstm/{v}/seed{s}/{f}" for v in ("gru_base", "lstm_b") for s in range(5)
                      for f in ("eval.json", "history.json")],
}


def files(study):
    for p in sorted((ROOT / "Algorytms").rglob("*")):
        if p.suffix in (".py", ".md") and "__pycache__" not in p.parts:
            yield p
    for d in ("Data/ML_dataset", "Data/Benchmark", "server"):
        yield from (p for p in sorted((ROOT / d).iterdir()) if p.is_file() and p.suffix != ".pyc")
    for f in ["requirements.txt"] + STUDY_FILES[study]:
        if (ROOT / f).exists():
            yield ROOT / f
        else:
            print(f"warning: {f} not found, not packed")


def main():
    study = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    out = ROOT / f"{study}_server_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files(study):
            name = p.relative_to(ROOT).as_posix()
            if p.suffix == ".sh":      # Linux line endings, whatever git did on Windows
                z.writestr(name, p.read_bytes().replace(b"\r\n", b"\n"))
            else:
                z.write(p, name)
        n = len(z.namelist())
    print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
