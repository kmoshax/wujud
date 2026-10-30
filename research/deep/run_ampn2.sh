#!/bin/sh
run() { .venv/bin/python -m deep.train "$@" 2>&1 | grep -v "Warning\|^  "; }
run --name cnn2d_aug_ampboth --encoder cnn2d --augment 1 --warp --inputs amp+ampn
run --name cnn2d_aug_ampboth_dann03 --encoder cnn2d --augment 1 --warp --inputs amp+ampn --dann 0.3
run --name cnn2d_aug05_ampn --encoder cnn2d --augment 0.5 --warp --inputs ampn
