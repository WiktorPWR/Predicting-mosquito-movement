#!/usr/bin/env bash
# LSTM vs GRU study, the whole thing on the Linux server (CPU only, low priority):
#   1. Python packages from requirements.txt into the venv (default ~/ml; created if missing)
#   2. training-step benchmark of this machine (once)            -> Results/lstm/bench_train_<host>.md
#   3. training of every variant x 5 seeds, PARALLEL runs at a time, 1 CPU thread each
#   4. evaluation on the benchmark, results table and figures     -> Results/lstm/results.md
#   5. Results/lstm packed into lstm_results.tar.gz (without checkpoints)
# Runs with nice 19 (shared server). Safe to stop at any time and to start again: finished runs are skipped and
# interrupted runs continue from their last epoch (checkpoint every epoch).
#
# Start (in the unpacked bundle folder):
#   tmux new -s lstm 'bash server/run_lstm_study.sh 2>&1 | tee -a lstm_study.log'
#   detach: Ctrl+B then D;  come back: tmux attach -t lstm
# Options (environment variables):  PARALLEL=4  VENV=~/other_venv  bash server/run_lstm_study.sh
# Progress:  tail -f lstm_study.log  |  ~/ml/bin/python -m Algorytms.lstm.run_all status
#            Results/lstm/<variant>/seed<k>/train.log (loss and time of every epoch)
set -euo pipefail
cd "$(dirname "$0")/.."
renice -n 19 -p $$ > /dev/null          # low priority for this script and everything it starts

VENV="${VENV:-$HOME/ml}"
PARALLEL="${PARALLEL:-$(nproc)}"
# memory guard: a training run needs ~0.45 GB, an evaluation worker up to ~0.6 GB; keep 1.5 GB free
avail_mb=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
max_par=$(( (avail_mb - 1500) / 600 ))
if [ "$max_par" -lt 1 ]; then max_par=1; fi
if [ "$PARALLEL" -gt "$max_par" ]; then
  echo "only $avail_mb MB RAM available: PARALLEL $PARALLEL -> $max_par"
  PARALLEL=$max_par
fi
echo "$(date '+%F %T') LSTM study on $(hostname): $PARALLEL parallel runs, $(nproc) cores, ${avail_mb} MB RAM available"

for f in Data/ML_dataset/train.npz Data/ML_dataset/val.npz Data/ML_dataset/norm.json \
         Data/Benchmark/_summary.csv Results/gru/disp_h32_k10/model.pt requirements.txt; do
  [ -e "$f" ] || { echo "missing $f - run this from the unpacked lstm_server_bundle.zip"; exit 1; }
done

# 1. environment
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "python venv failed (Debian: sudo apt install python3-venv)"; exit 1; }
fi
"$VENV/bin/pip" install -q -r requirements.txt
P="$VENV/bin/python"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$P -c "import torch, numpy, pandas, matplotlib; print('torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| cuda', torch.cuda.is_available())"

# GRU baseline seed 0 = the selected GRU (the training code reproduces it exactly)
mkdir -p Results/lstm/gru_base/seed0
[ -e Results/lstm/gru_base/seed0/model.pt ] || cp Results/gru/disp_h32_k10/model.pt Results/gru/disp_h32_k10/history.json Results/lstm/gru_base/seed0/

# 2. benchmark of this machine (a few minutes, once)
if [ ! -e "Results/lstm/bench_train_$(hostname).md" ]; then
  OMP_NUM_THREADS=$(nproc) $P -m Algorytms.lstm.bench_train --threads "1,$(nproc)" --save
fi

# 3. training
$P -m Algorytms.lstm.run_all --parallel "$PARALLEL" --longest-first
$P -m Algorytms.lstm.run_all status

# 4. evaluation, table, figures
$P -m Algorytms.lstm.run --jobs "$PARALLEL"

# 5. results to take home
tar czf lstm_results.tar.gz --exclude='checkpoint.pt' Results/lstm
echo "$(date '+%F %T') finished - copy lstm_results.tar.gz back and unpack it in the repository folder"
