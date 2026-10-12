# Predicting-mosquito-movement
This is a comparison of classical mathematical and machine learning-based methods for predicting the movement of Aedes aegypti mosquitoes.

## Project structure

| folder | content |
|---|---|
| [Data/](Data/README.md) | data pipeline: splitting, stitching tracker fragments, gap filling, filtering, benchmark |
| [Algorytms/](Algorytms/README.md) | prediction algorithms and the shared evaluation protocol |
| [Algorytms/kalman/](Algorytms/kalman/README.md) | Kalman filter (constant velocity) |
| [Algorytms/particle/](Algorytms/particle/README.md) | particle filter |
| [Algorytms/baselines/](Algorytms/baselines/README.md) | constant velocity, linear autoregression |
| [Algorytms/gru/](Algorytms/gru/README.md) | GRU networks |
| [Results/comparison/](Results/comparison/comparison_table.md) | all models compared on the benchmark |
| `Results/` | tables and figures of every algorithm, interactive 3D viewer (`Results/viewer/index.html`) |
