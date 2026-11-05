# link_net v5: small bedrooms (2026-11-05)

The location network (`tools/link_net.py`) was trained on rooms of 3-5.5 by 3-5 m. The taped bedroom
(`rooms/home_20261105.json`) is 2.7 by 2.4 m. New simulated data, `export_tracking ... small`:
480 bedrooms of 2.3-3.3 by 2.2-3 m, four boards in their corners in every other room, every link
measured both ways like the real boards (`cache/tracking/small{3,4}_p*.jsonl`, seeds 3,000,000+).
Both directions of a pair are averaged into one link before training and scoring, as the live view
does (`crates/app/src/link_net.rs: add_second`).

v5 = v3 fine-tuned 16 epochs at lr 2e-4 on v3_part0-2, v3four_p0-5 and the small rooms
(`cache/tracking/link_net_v5.log`). Median location error while walking, on the last 20% of rooms of
each file (never trained on by either model):

| rooms | boards | v3 | v4 (directions not averaged) | v5 | room's centre |
|---|---|---|---|---|---|
| v3four_p5 (3-5.5 m) | 4 | 0.57 | 0.57 | 0.57 | 0.91 |
| v3_part2 (3-5.5 m) | 3 | 0.67 | 0.67 | 0.66 | 0.81 |
| small4_p0-3 | 4 | 0.38-0.45 | 0.30-0.33 | 0.30-0.31 | 0.44-0.46 |
| small3_p0-1 | 3 | 0.45-0.48 | 0.36-0.38 | 0.34-0.35 | 0.43-0.45 |

In a small room v3 was hardly better than the room's centre, and with three boards worse; v5 is a
quarter better than v3 there, and the same in large rooms. Simulation only: the real room has not been
scored, since nobody's position was recorded.
