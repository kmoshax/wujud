#!/bin/sh
# compile RuView's edge_processing.c unchanged, with RuView's own host-test stubs
R=${RUVIEW:-$HOME/Projects/ext/RuView}/firmware/esp32-csi-node
cc -std=c99 -O2 -Wall -Wno-unused-function -I"$R/test/stubs" -I"$R/main" replay.c "$R/test/stubs/esp_stubs.c" -o replay -lm
