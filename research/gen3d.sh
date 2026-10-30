#!/bin/bash
# 3-D hall datasets used in results3d (~40 min on 2 cores)
set -e
cd "$(dirname "$0")"
python3 -m hallsense.dataset3d --out data3/test  --halls 20 --scenes 2 --seed 12 --nodes 12
python3 -m hallsense.dataset3d --out data3/train --halls 30 --scenes 2 --seed 11 --nodes 6,8,10,12
python3 -m hallsense.dataset3d --out data3/sweep --halls 6 --seed 13 --node-sweep 4,6,8,12
echo DONE3D
