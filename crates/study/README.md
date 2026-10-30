# wujud-study

Simulation studies built on `wujud-sim` and `wujud-model`. Currently one: `limits_study`, "which physical
change breaks the single-link limit of Wi-Fi presence sensing?". Protocol, results, physics assumptions and their
sources: `research/results/limits/README.md`.

| Module | What it does |
|---|---|
| `configs` | the compared hardware configurations (band, subcarrier layout, hopping, packet rate, boards, mesh, placement) and their extra cost |
| `scenes` | per room seed: one 60-s session per activity and a ~10-min day, simulated with the calibrated realism knobs and reduced to window features right away |
| `windows` | window features and static profiles of one link; with channel hopping, per-channel grids side by side (one channel = exactly `features::windows` + `profile_windows`, tested) |
| `evaluate` | per-configuration classifier (`train_classifier`), still-vs-empty AUC, 5 % CFAR breathing visibility and rate error, motion recall, the occupancy-latch protocol of `research/bench/occupancy_sim.py`, room-level bootstrap errors |

```
cargo run --release -p wujud-study --bin limits_study -- --configs baseline,mesh2 --train-rooms 300 --test-rooms 200
```
Results go to `research/results/limits/config_<name>.json`; existing files are skipped unless `--force`.
