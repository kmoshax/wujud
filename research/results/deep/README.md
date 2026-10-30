# Deep learning on raw CSI, trained on simulation: can it tell a still person from an empty room on real ESP32 data?

**Short answer: no.** The model chosen on the selection split scores a holdout window AUC of **0.655 (95 % CI 0.37–0.91)**. The hand-feature baseline scores 0.651. The paired difference is **−0.23 to +0.22**, so the two cannot be told apart. The deep model also detects motion less well on holdout (recall 0.88 and 4.2 % false alarms, against 1.00 and 0 %). Nothing is exported for the Rust port.

## Setup

* **Simulated training data**: `crates/app/examples/export_sim_windows.rs` and `deep/build_cache.py` produce it.
  * 3000 random bedrooms. Each room has five 60 s single-activity sessions (away, lie, sit, stand, walk), making 15 000 sessions. One router → board link.
  * Every session is randomised:
    * packet rate U(50, 120)/s and harshness U(0.8, 1.6);
    * realism knobs spanning the original simulator to the calibrated one: rcs_scale log-U(0.1, 1), breath_scale U(0.5, 1), evm_shift U(0, 6) dB;
    * a distractor (fan, pet, vacuum or door) in 20 % of sessions.
* **Preprocessing** (`deep/preprocess.py`) is identical for simulation and real data:
  * Both sources are mapped to the 52 L-LTF data subcarriers (k = ±1..±26). The simulator reports HT-LTF 56. Real ESP32 buffers come in two layouts, FFT order or centred; the layout is detected per recording and checked by spectral smoothness. The dead k=+1 bin of the original ESP32 is filled from its neighbours.
  * Per-packet amplitude is divided by its per-packet mean to remove AGC, then converted to dB.
  * Phase is sanitised by removing a·k + b + c·[k>0].
  * RSSI is kept as a separate channel.
  * Everything is resampled onto a 20 Hz grid and cut into 20 s windows with a 5 s hop. The window counts match the baseline exactly: 1011 selection, 1044 holdout.
* **Inputs** are always relative to the window itself:
  * `amp`: dB minus its temporal mean.
  * `ampn`: the same, divided by the window's receiver jitter (median over subcarriers of the sample-to-sample std).
  * Optional extras: sanitised phase, RSSI, and the static subcarrier profile.
* **Models** (`deep/models.py`):
  * Encoders: a 2-D CNN over time × subcarrier, a 1-D CNN with subcarriers as channels, a temporal transformer (1 s tokens) and a spectral CNN (log power spectrum per subcarrier).
  * Heads: 3 classes (empty, still, moving) and a dedicated still-vs-empty logit.
* **Sim-to-real methods** (`deep/train.py`, `deep/augment.py`):
  * plain simulation training;
  * augmentation: frequency-selective gain drift, AGC steps, amplitude-dependent quantisation noise, packet-loss holds, subcarrier mirror, time warp ×0.8–1.25;
  * DANN: gradient reversal against UNLABELED real windows from the train split;
  * SimCLR contrastive pretraining on unlabeled real-train windows plus simulation, then fine-tuning on simulation labels.
  * Real labels are never used.
* **Protocol**:
  * Variants are chosen only on selection. The rule is: highest selection window AUC among variants with selection motion false alarm ≤ 2 %.
  * Holdout was scored once, after the selection phase was frozen, for every variant (`deep/report.py`).
  * The 95 % CIs come from 2000 bootstrap resamples of recordings within each class. Differences against the baseline are paired, on the same resamples.

## Results (still-vs-empty AUC; recording AUC = mean score over the first minute, as in `bench/phase_exp.py`)

| model | sel. window AUC [95 % CI] | sel. rec. | hold. window AUC [95 % CI] | hold. rec. [CI] | hold. Δ vs baseline [CI] | hold. motion recall / FA |
|---|---|---|---|---|---|---|
| Hand features + GBDT (baseline) | 0.546 [0.31, 0.77] | 0.661 | 0.651 [0.42, 0.86] | 0.688 [0.35, 0.96] | — | 1.00 / 0.000 |
| **2-D CNN, aug, noise-normalised, DANN 0.3, seed 1 (chosen)** | **0.684** [0.47, 0.84] | 0.679 | **0.655** [0.37, 0.91] | 0.708 [0.42, 0.96] | [−0.23, +0.22] | 0.88 / 0.042 |
| same, seed 0 | 0.678 | 0.750 | 0.648 | 0.667 | | 0.87 / 0.068 |
| same, seed 2 | 0.658 | 0.696 | 0.688 | 0.729 | | 0.98 / 0.023 |
| same, 3-seed ensemble | 0.677 [0.46, 0.84] | 0.732 | 0.665 [0.38, 0.92] | 0.729 | [−0.23, +0.26] | 0.93 / 0.045 |
| 2-D CNN, aug, noise-normalised, DANN 1.0 | 0.663 | 0.696 | 0.664 | 0.646 | | 1.00 / 0.049 |
| 2-D CNN, aug, raw + noise-normalised, DANN 0.3 | 0.660 | 0.714 | 0.689 | 0.729 | | 0.92 / 0.003 |
| 2-D CNN, aug, noise-normalised, contrastive pretrain | 0.635 | 0.661 | 0.694 | 0.750 | | 1.00 / 0.178 |
| 2-D CNN, aug, noise-normalised | 0.605 | 0.589 | 0.700 | 0.771 | | 1.00 / 0.240 |
| 2-D CNN, plain (raw amp) | 0.509 | 0.518 | 0.642 | 0.688 | | 0.98 / 0.025 |
| 2-D CNN, aug | 0.448 | 0.518 | 0.737 | 0.792 | | 0.94 / 0.028 |
| 2-D CNN, aug, DANN 0.3 | 0.534 | 0.661 | 0.720 | 0.708 | | 0.82 / 0.012 |
| 2-D CNN, aug, contrastive pretrain | 0.457 | 0.571 | 0.695 | 0.667 | | 0.98 / 0.014 |
| 2-D CNN, aug, +phase | 0.418 | 0.500 | 0.749 | 0.812 | | 0.88 / 0.009 |
| 2-D CNN, aug, +RSSI | 0.287 | 0.464 | 0.829 | 0.833 | | 0.98 / 0.005 |
| 2-D CNN, aug, +static profile | 0.492 | 0.589 | 0.766 | 0.812 | | 1.00 / 0.003 |
| 1-D CNN plain / aug | 0.443 / 0.411 | | 0.732 / 0.766 | | | |
| Transformer plain / aug / aug+DANN / aug+SSL | 0.426 / 0.421 / 0.464 / 0.387 | | 0.623 / 0.625 / 0.619 / 0.684 | | | |
| Spectral CNN plain / aug / aug+noise-norm. | 0.472 / 0.436 / 0.548 | | 0.593 / 0.532 / 0.540 | | | |
| *Reference: receiver noise level only (no training)* | 0.349 [0.08, 0.70] | 0.357 | **0.844** [0.62, 1.00] | 0.875 | | — |

All the numbers and per-device breakdowns are in `summary.json`. The per-run configurations and scores are in `runs/*.json`.

## What the numbers say

* **The holdout split rewards receiver noise, not people.** Ranking holdout recordings by the receiver's jitter alone, with no model at all, gives an AUC of 0.844. Several holdout still-person recordings were made at RSSI −70 to −82 dBm, which is noisy, while the holdout empty recordings are quiet night runs. On selection the same trick scores 0.349, so the relationship is reversed. Every variant that lets noise or RSSI through therefore looks good on holdout and bad on selection; adding RSSI as an input scores 0.287 on selection and 0.829 on holdout. Across all variants, the selection and holdout AUCs are not positively related (Spearman −0.21). A holdout "win" by any of those variants would have been an artefact.
* **What helped on selection:**
  * *Noise normalisation* is the largest single gain (cnn2d with augmentation: 0.448 → 0.605).
  * *DANN* against unlabeled real windows gives the next gain (0.605 → 0.678). It also fixes the motion false alarms that noise normalisation introduces (24 % → 0.1 % on selection).
  * Three seeds agree on selection (0.658–0.684).
  * Even so, the gain is not significant: the paired selection Δ against the baseline is [−0.06, +0.33].
* **What did not help:**
  * Plain simulation training: every encoder sat at or below chance on selection (0.43–0.51), even though the simulation-validation AUC was 0.83–0.88.
  * Augmentation alone.
  * Contrastive pretraining, with or without noise normalisation.
  * Transformers and spectral inputs.
  * Phase, RSSI and the static-profile channels, which all hurt on selection.
* **Holdout:** the chosen model and its ensemble land at 0.655–0.665, the same as the baseline. Within each device (`fig2_per_device.png`) the deep model is above the baseline on all three holdout devices. But each device has only one empty recording, so this cannot carry a claim. On one selection device (4ae708) it collapses to 0.05.

## Verdict

Deep models on raw CSI, trained only on the simulator with augmentation, DANN or contrastive pretraining on unlabeled real data, **do not beat the single-link still-vs-empty limit**. The chosen model scores 0.655 on holdout against the baseline's 0.651. The CIs are very wide (0.37–0.91) because there are only 4 empty and 12 still holdout recordings, and the paired Δ CI includes 0. The split itself is confounded by receiver noise and time of day (`fig3_time_of_day.png`): a detector that measures nothing but noise beats every model on holdout. More labelled real sessions with matched day and night empty and still recordings would be needed before any of these numbers mean anything. No weights or ONNX file are exported.

## Files

* Code:
  * `crates/app/examples/export_sim_windows.rs`
  * `research/deep/{preprocess,build_cache,data,augment,models,train,evaluate,report,baseline_scores}.py`
  * `research/deep/run_*.sh` (the exact runs, in order)
  * `research/tools/deep_figures.py`
* Results: `summary.json`, `runs/*.json`, `baseline_check.json` (the baseline re-scored, identical to `phase_exp.json`)
* Figures:
  * `fig1_auc_variants.png`: selection and holdout AUC with CIs for every variant.
  * `fig2_per_device.png`
  * `fig3_time_of_day.png`: the first-minute score of each holdout recording against the hour it started. Hollow markers are long night recordings.
* Caches (regenerate with `deep.build_cache`, about 10 min): `research/cache/deep/` (sim/real streams, model weights and scores)

Reproduce: `.venv/bin/python -m deep.build_cache 3000`, `.venv/bin/python -m deep.baseline_scores`, `sh deep/run_plain.sh` … `run_ampn3.sh`, `.venv/bin/python -m deep.report`, `.venv/bin/python tools/deep_figures.py`. Run all of these from `research/`.
