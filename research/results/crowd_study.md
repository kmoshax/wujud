# Several people in one room: how far the links go

Trained on the simulator only, measured on rooms never seen in training (last 20 % of each file).
Data: `export_crowd` (180 s rooms, 0-5 people each on their own walk/sit/stand/lie/away plan, 50 packets/s,
harsh 0.7-2.0). 600 rooms with 3 devices, 379 with 4, 318 with 6, 61 with 8; 275 test rooms, about 50 000
test seconds. Model: `tools/crowd_net.py` (one network for every device count), raw numbers in `crowd_net.json`.

## Counting, per number of people actually in the room

Exact accuracy (within ±1 in brackets). "Walking" is how many of them walk, counted in the same seconds.
Always guessing the training set's most common count scores 20-25 % (present) and 39-51 % (walking) overall.

| devices (links) | 0 | 1 | 2 | 3 | 4 | 5 | all seconds |
|---|---|---|---|---|---|---|---|
| 3 (3) present | 93 % | 32 (75) | 35 (92) | 58 (96) | 25 (80) | 1 (40) | 46 % (85) |
| 3 walking | 99 | 78 | 69 | 53 | 40 | 35 | 69 % |
| 4 (6) present | 93 | 59 (90) | 45 (95) | 50 (90) | 21 (81) | 1 (45) | 58 % (90) |
| 4 walking | 100 | 87 | 69 | 58 | 56 | 36 | 77 % |
| 6 (15) present | 95 | 76 (93) | 32 (94) | 48 (91) | 19 (72) | 6 (47) | 58 % (90) |
| 6 walking | 100 | 91 | 79 | 74 | 49 | 46 | 82 % |
| 8 (28) present* | 99 | 90 (100) | 35 (99) | 26 (72) | 18 (53) | 1 (63) | 54 % (87) |

## Placing every walker, per number walking at once

Median per-person error after optimal (Hungarian) matching, given the true number of walkers.
Centre: everyone placed at the room centre. Chance: as many random spots as walkers, also matched.

| devices | 1 walker | 2 | 3 | 4 | centre | chance (1 / 3 walkers) |
|---|---|---|---|---|---|---|
| 3 | 0.80 m | 0.80 | 0.73 | 0.66 | 0.85-0.89 | 1.39 / 0.98 |
| 4 | 0.59 | 0.67 | 0.67 | 0.58 | 0.81-0.92 | 1.34 / 1.07 |
| 6 | 0.46 | 0.60 | 0.60 | 0.58† | 0.82-0.90 | 1.35 / 1.04 |
| 8* | 0.45 | 0.56 | 0.59 | - | 0.86-0.93 | 1.40 / 1.06 |

Walking count right: 1 walker 71/73/86/92 %, 2 walkers 43/52/65/58 %, 3 walkers 21/19/24/13 %, 4+ walkers 0 %
(3/4/6/8 devices). \* 8 devices: only 12 test rooms. † 63 seconds.

## Where it breaks, and why

1. **Still people.** An empty room is right 93-99 % of the time, but with two or more people in the room
   the presence count is right only within ±1, while the walking count in the same seconds is far better
   (e.g. 6 devices, 3 people: 48 % present vs 74 % walking). Every link feature is
   relative to the link's own quiet level (there is no empty-room reference), and a still body changes the
   channel by a static amount, not a moving one. More links do not fix it: the exact presence count stays at
   54-58 % from 4 to 8 devices. Only memory (seeing someone come in and not leave) or a reference could.
2. **Counting walkers stops at two.** One walker against none is solved (86-92 % with 6-8 devices). Two are
   counted right 52-65 % of the time, three about 20 %, four never, with any number of devices. The RSSI spread
   grows like the square root of the number of walkers (median 1.09, 1.52, 1.92, 2.09 dB for 1-4 walkers): the
   extra signal each newcomer adds shrinks, and the field from people near the same links adds up, not apart.
3. **Placing walkers is limited by links per unknown.** Each walker has 2 unknowns plus a strength. Three
   devices give 3 links: fewer measurements than unknowns even for one person, so 0.80 m is barely better than
   the centre (0.86). Counting which links' sensitive zones (extra path < 0.25 m) cover a spot, 3 links split
   the floor into about 4 distinguishable cells (53 % covered by none, two people 0.6 m+ apart share a cell 42 %
   of the time); 4 devices about 11 cells (26 % shared), 6 devices 37 (7 %), 8 devices 75 (1 %). That is why
   one walker goes 0.80 → 0.59 → 0.46 m, then stops at 0.45 m with 8 devices: once zones overlap more finely
   than a body (0.5 m), more links measure the same body twice. With several walkers the error rises to
   0.56-0.67 m and stays there; with 3-4 walkers it is helped by matching (chance itself falls to about 1 m).
4. **Five people is past every limit tested**: presence 1-6 % exact.

## Not verified

Simulator only; real boards were never tried with more than one person. People in the simulator appear and
leave without walking through a door, so entering and leaving cannot be learnt as events. Two people are kept
0.6 m apart by smoothing, so brief closer passes remain. Device placement is random mesh, never "smart".
