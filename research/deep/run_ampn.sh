#!/bin/sh
run() { .venv/bin/python -m deep.train "$@" 2>&1 | grep -v "Warning\|^  "; }
run --name cnn2d_plain_ampn --encoder cnn2d --inputs ampn
run --name cnn2d_aug_ampn --encoder cnn2d --augment 1 --warp --inputs ampn
run --name transformer_aug_ampn --encoder transformer --augment 1 --warp --inputs ampn
run --name spectral_aug_ampn --encoder spectral --augment 1 --warp --inputs ampn
run --name cnn2d_aug_ampn_dann03 --encoder cnn2d --augment 1 --warp --inputs ampn --dann 0.3
run --name cnn2d_aug_ampn_ssl --encoder cnn2d --augment 1 --warp --inputs ampn --ssl 2000
