#!/bin/bash
# Full simulated dataset used for the results in the report (~1 h on 2 cores)
set -e
cd "$(dirname "$0")"
python3 -m hallsense.dataset --out data/train --halls 100 --scenes 4 --seed 1
python3 -m hallsense.dataset --out data/test  --halls 30 --scenes 2 --seed 2 --nodes 12
python3 -m hallsense.dataset --out data/sweep --halls 12 --seed 3 --node-sweep 6,8,12,16,24 --placement perimeter
python3 -m hallsense.dataset --out data/harsh --halls 15 --scenes 1 --seed 4 --nodes 12 --harsh 1.6
python3 -m hallsense.dataset --out data/calm  --halls 15 --scenes 1 --seed 5 --nodes 12 --restless 0.4
python3 -m hallsense.dataset --out data/train_rows --halls 40 --scenes 4 --seed 6 --nodes 10,12,14,16 --placement rows
python3 -m hallsense.dataset --out data/test_rows  --halls 30 --scenes 2 --seed 2 --nodes 12 --placement rows
python3 -m hallsense.dataset --out data/hallcal --halls 25 --scenes 4 --seed 8 --nodes 12 --shared-env
echo DONE2
