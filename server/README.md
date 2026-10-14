# Running the LSTM vs GRU study on the server

Target: the "sandbox" server.

| item | value |
|---|---|
| CPU | Debian in an Incus container, i5-8500, 6 cores, no CUDA |
| RAM | 7.6 GB |
| Python | venv in `~/ml` |
| access | SSH, shared with others (hence low priority) |

Everything runs on the CPU.

## 1. On this computer: pack

```
.venv/Scripts/python -m server.make_bundle
```

This creates `lstm_server_bundle.zip` in the repository root (~34 MB). The data are not in git, so a clone is not
enough; the zip contains:
- the code and `requirements.txt`;
- `Data/ML_dataset` and `Data/Benchmark`;
- the selected GRU, which is GRU baseline seed 0;
- the results already computed here.

## 2. On the server: unpack and start

```
scp lstm_server_bundle.zip sandbox:~
ssh sandbox
mkdir -p ~/lstm_study && cd ~/lstm_study && unzip -o ~/lstm_server_bundle.zip
tmux new -s lstm 'bash server/run_lstm_study.sh 2>&1 | tee -a lstm_study.log'
```

Detach with `Ctrl+B`, then `D`, and log out; the run continues. Come back later with `tmux attach -t lstm`.

[run_lstm_study.sh](run_lstm_study.sh) does everything in order:

1. **Packages:** `pip install -r requirements.txt` into `~/ml`; torch from the CPU-only index.
2. **Benchmark:** the training-step benchmark of the server, once, a few minutes →
   `Results/lstm/bench_train_<host>.md`.
3. **Training:** 34 runs (7 variants × 5 seeds, minus GRU baseline seed 0, which is reused).
   - 6 at a time, 1 thread each, slowest runs first.
   - Each run takes ~0.45 GB RAM.
   - The script lowers the number of parallel runs if less RAM is free, so it never falls into swap.
4. **Evaluation:** all models on the benchmark, 6 in parallel → `Results/lstm/results.md` + figures.
5. **Packing:** `lstm_results.tar.gz` with `Results/lstm`, without checkpoints.

Everything runs with `nice 19`. Why 1 thread per run rather than 1 run with 6 threads: with networks of a few
thousand parameters, extra threads make one step slower, not faster. The laptop benchmark shows it: GRU 9.6 ms
per step with 1 thread, 22.8 ms with 4.

**Expected time:** about 10–12 h. The benchmark in step 2 gives the exact per-epoch numbers for the server.

## Stopping and resuming

- **Stop:** `Ctrl+C` in tmux, or `tmux kill-session -t lstm`.
- **Resume:** start the same `tmux new ...` command again.
  - Finished runs are skipped.
  - Interrupted runs continue after their last finished epoch: a checkpoint is saved after every epoch.
  - Already evaluated models are not evaluated again.

## Progress

```
~/ml/bin/python -m Algorytms.lstm.run_all status      # every run: waiting / epoch k of 40 / done
tail -f lstm_study.log
tail Results/lstm/lstm_c100/seed0/train.log           # loss, validation error and time of every epoch
```

Partial table at any time, on the finished runs:
`OMP_NUM_THREADS=1 nice -n 19 ~/ml/bin/python -m Algorytms.lstm.run --jobs 1`.

## 3. Back on this computer

Copy `lstm_results.tar.gz` back and unpack it in the repository root:

```
scp sandbox:~/lstm_study/lstm_results.tar.gz .
tar xzf lstm_results.tar.gz
```

This fills `Results/lstm/`.

The Raspberry Pi time column is filled after running
`python -m Algorytms.lstm.bench_cpu --save Results/lstm/timing_raspberry.json` on the Pi, then
`python -m Algorytms.lstm.run` once more.
