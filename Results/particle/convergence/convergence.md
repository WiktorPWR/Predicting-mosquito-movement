# From which sample on are predictions sensible?

Every model was started from scratch 961 times (every 1 s of every test trajectory) and fed only the frames from that point on. *Sensible from sample k* = first k after which the mean error stays within 15% of its steady-state value (mean over samples 201-300).

| model | horizon | sensible from sample | = time after start | error at sample 1 [cm] | steady-state error [cm] |
|---|---|---|---|---|---|
| Particle filter (N=5000, keep 100%, 0.504 m/s, likelihood 0.2 cm) | 0.01 s | 2 | 0.02 s | 0.57 | 0.22 |
| Particle filter (N=5000, keep 100%, 0.504 m/s, likelihood 0.2 cm) | 0.1 s | 2 | 0.02 s | 4.91 | 2.05 |
| Particle filter (N=5000, keep 100%, 0.504 m/s, likelihood 0.2 cm) | 0.5 s | 2 | 0.02 s | 23.49 | 13.88 |
| Particle filter (N=5000, keep 100%, 0.504 m/s, likelihood 0.2 cm) | 1 s | 2 | 0.02 s | 43.40 | 34.62 |
| Kalman CV (0.336 m/s) | 0.01 s | 2 | 0.02 s | 0.57 | 0.17 |
| Kalman CV (0.336 m/s) | 0.1 s | 2 | 0.02 s | 4.91 | 1.76 |
| Kalman CV (0.336 m/s) | 0.5 s | 2 | 0.02 s | 23.49 | 13.09 |
| Kalman CV (0.336 m/s) | 1 s | 2 | 0.02 s | 43.40 | 33.60 |
