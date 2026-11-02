# Body movements from simulation only

Classes: still (holding any posture, or away), walk, wave (hand raised, waving or held), bow (ruku'), prostrate (sujud), sit_down, stand_up. Going into and out of a bow, prostration or wave counts as that movement. Scored per second.

## Data

`crates/app/examples/export_movements.rs`: 2000 random rooms × 240+ s, 3 or 4 devices in a full mesh (3 or 6 links), random board places, harshness 0.7–2.0, person RCS ×0.1–1, EVM +0–6 dB, a fan, pet or door in 15% of rooms. Plans: `random_movement_plan` (prayer cycles, standing and seated waves, sit/stand cycles, walking, lying, leaving). Split by room seed: 1600 train, 200 validation (epoch and settings chosen here), 200 test (read once). Stress set: 200 more rooms with harshness 2.0–3.5, beyond anything trained on.

## Unseen rooms (test, read once)

`research/tools/movement_net.py`, scale=1, 12 epochs, epoch picked on validation. Full matrices in `unseen_rooms.txt`.

| class | recall | precision |
|---|---|---|
| still | 82.2% | 92.1% |
| walk | 88.6% | 94.8% |
| wave | 82.9% | 60.4% |
| bow | 87.3% | 81.2% |
| prostrate | 89.7% | 89.3% |
| sit_down | 75.7% | 43.9% |
| stand_up | 84.3% | 57.4% |

- Mean recall of the five movement classes: 84.0%. Accuracy: 85.6%.
- 3 devices: 81.9%. 4 devices: 86.8%.
- Per room accuracy: worst 58%, 10th percentile 77%, median 87%.
- Stress set (harsher hardware): 77.9% movement recall.
- Baseline (`movement_baseline.py`: motion energy pooled over links and 1/3/7 s, gradient-boosted trees): 53.1% movement recall, accuracy 50.8%.

## What changed along the way

- First run (`first_run_old_labels.txt`): the walk to the chair and sitting down happened together and were all labelled sit_down. Walk was then read as sit_down 1822 times. Now, in movement sessions, the simulator walks first and then sits down (and stands up before walking off). Validation went from 82.2% to 85.4%.
- Dividing each link by its own typical 1 s change helped a little (`ablation_no_link_scale_old_labels.txt`: 80.5% against 81.6% balanced, on the old labels).

## Limits

- Simulation only. No real recording of these movements has been scored.
- The answer comes about 6 s late: the network reads ±6.2 s around each second.
- Wave and the transitions raise false alarms. About 7% of still seconds are called wave, so wave precision is 60%.
- "still" does not tell standing, sitting, kneeling, lying or an empty room apart.
