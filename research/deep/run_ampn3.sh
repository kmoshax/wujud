#!/bin/sh
run() { .venv/bin/python -m deep.train "$@" 2>&1 | grep -v "Warning\|^  "; }
run --name cnn2d_aug_ampn_dann03_s1 --encoder cnn2d --augment 1 --warp --inputs ampn --dann 0.3 --seed 1
run --name cnn2d_aug_ampn_dann03_s2 --encoder cnn2d --augment 1 --warp --inputs ampn --dann 0.3 --seed 2
run --name cnn2d_aug_ampn_dann1 --encoder cnn2d --augment 1 --warp --inputs ampn --dann 1.0
run --name transformer_aug_ampn_dann03 --encoder transformer --augment 1 --warp --inputs ampn --dann 0.3
