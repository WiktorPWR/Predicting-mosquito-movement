# Predicting-mosquito-movement
This is a comparison of classical mathematical and machine learning-based methods for predicting the movement of Aedes aegypti mosquitoes.

**All tested algorithms, their variants and results in one place: [ALGORITHMS.md](ALGORITHMS.md).**

## Project structure

| folder | content |
|---|---|
| [Data/](Data/README.md) | data pipeline: splitting, stitching tracker fragments, gap filling, filtering, benchmark |
| [Algorytms/](Algorytms/README.md) | prediction algorithms and the shared evaluation protocol |
| [Algorytms/kalman/](Algorytms/kalman/README.md) | Kalman filter (constant velocity) |
| [Algorytms/particle/](Algorytms/particle/README.md) | particle filter |
| [Algorytms/baselines/](Algorytms/baselines/README.md) | constant velocity, linear autoregression |
| [Algorytms/gru/](Algorytms/gru/README.md) | GRU networks |
| [Algorytms/lstm/](Algorytms/lstm/README.md) | LSTM vs GRU study (5 seeds per variant) |
| [Algorytms/transformer/](Algorytms/transformer/README.md) | small transformers (5 seeds per variant) |
| [server/](server/README.md) | running long trainings on the server |
| [Results/benchmark/](Results/benchmark/README.md) | **final benchmark: the best configuration of every algorithm** |
| [Results/comparison/](Results/comparison/comparison_table.md) | all models compared on the benchmark |
| `Results/` | tables and figures of every algorithm, interactive 3D viewer (`Results/viewer/index.html`) |
