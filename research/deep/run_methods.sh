#!/bin/sh
run() { .venv/bin/python -m deep.train "$@" 2>&1 | grep -v "Warning\|^  "; }
run --name transformer_plain --encoder transformer
for enc in cnn2d transformer cnn1d spectral; do run --name ${enc}_aug --encoder $enc --augment 1 --warp; done
run --name cnn2d_aug_dann03 --encoder cnn2d --augment 1 --warp --dann 0.3
run --name cnn2d_aug_dann1 --encoder cnn2d --augment 1 --warp --dann 1.0
run --name cnn2d_aug_ssl --encoder cnn2d --augment 1 --warp --ssl 2000
run --name transformer_aug_dann03 --encoder transformer --augment 1 --warp --dann 0.3
run --name transformer_aug_ssl --encoder transformer --augment 1 --warp --ssl 2000
run --name cnn2d_aug_phase --encoder cnn2d --augment 1 --warp --inputs amp+phase
run --name cnn2d_aug_rssi --encoder cnn2d --augment 1 --warp --inputs amp+rssi
run --name cnn2d_aug_static --encoder cnn2d --augment 1 --warp --inputs amp+static
run --name cnn2d_aug_all --encoder cnn2d --augment 1 --warp --inputs amp+phase+rssi+static
