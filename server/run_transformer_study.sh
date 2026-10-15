#!/usr/bin/env bash
# Transformer study, the whole thing on the Linux server (CPU only, low priority):
#   1. Python packages from requirements.txt into the venv (default ~/ml; created if missing)
#   2. training of the 3 transformer variants x 5 seeds, PARALLEL runs at a time, 1 CPU thread each, slowest first
#   3. evaluation on the benchmark + comparison with the GRU baseline and LSTM B -> Results/transformer/results.md
#   4. Results/transformer packed into transformer_results.tar.gz (without checkpoints)
# Runs with nice 19 (shared server). Safe to stop at any time and to start again: finished runs are skipped and
# interrupted runs continue from their last epoch (checkpoint every epoch).
#
# Start (in the unpacked bundle folder):
#   tmux new -s tf 'bash server/run_transformer_study.sh 2>&1 | tee -a transformer_study.log'
#   detach: Ctrl+B then D;  come back: tmux attach -t tf
# Options (environment variables):  PARALLEL=4  VENV=~/other_venv  bash server/run_transformer_study.sh
# Progress:  tail -f transformer_study.log  |  ~/ml/bin/python -m Algorytms.transformer.run_all status
#            Results/transformer/<variant>/seed<k>/train.log (loss and time of every epoch)
set -euo pipefail
cd "$(dirname "$0")/.."
renice -n 19 -p $$ > /dev/null          # low priority for this script and everything it starts

VENV="${VENV:-$HOME/ml}"
PARALLEL="${PARALLEL:-$(nproc)}"
# memory guard: a training run needs ~0.5 GB, an evaluation worker up to ~0.6 GB; keep 1.5 GB free
avail_mb=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
max_par=$(( (avail_mb - 1500) / 600 ))
if [ "$max_par" -lt 1 ]; then max_par=1; fi
if [ "$PARALLEL" -gt "$max_par" ]; then
  echo "only $avail_mb MB RAM available: PARALLEL $PARALLEL -> $max_par"
  PARALLEL=$max_par
fi
echo "$(date '+%F %T') transformer study on $(hostname): $PARALLEL parallel runs, $(nproc) cores, ${avail_mb} MB RAM available"

for f in Data/ML_dataset/train.npz Data/ML_dataset/val.npz Data/ML_dataset/norm.json Data/Benchmark/_summary.csv \
         Results/lstm/gru_base/seed0/eval.json Results/lstm/lstm_b/seed0/eval.json requirements.txt; do
  [ -e "$f" ] || { echo "missing $f - run this from the unpacked transformer_server_bundle.zip"; exit 1; }
done

# 1. environment
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "python venv failed (Debian: sudo apt install python3-venv)"; exit 1; }
fi
"$VENV/bin/pip" install -q -r requirements.txt
P="$VENV/bin/python"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$P -c "import torch, numpy, pandas, matplotlib; print('torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__)"

# 2. training
$P -m Algorytms.transformer.run_all --parallel "$PARALLEL"
$P -m Algorytms.transformer.run_all status

# 3. evaluation, comparison table, figures
$P -m Algorytms.transformer.run --jobs "$PARALLEL"

# 4. results to take home
tar czf transformer_results.tar.gz --exclude='checkpoint.pt' Results/transformer
echo "$(date '+%F %T') finished - copy transformer_results.tar.gz back and unpack it in the repository folder"
