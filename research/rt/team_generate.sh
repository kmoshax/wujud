#!/usr/bin/env bash
# Generate a range of worlds with several workers on one NVIDIA GPU, each with a live view.
#   rt/team_generate.sh <first world> <how many> [workers=4] [out=cache/rt]
# World N is identical on every machine: give each teammate a different range, then gather the .npz files.
# Run it in its own memory-capped unit, so a runaway never takes the desktop down with it:
#   systemd-run --user --collect -p MemoryMax=10G -p MemorySwapMax=1G rt/team_generate.sh 0 500
set -euo pipefail
first=$1; count=$2; workers=${3:-4}; out=${4:-cache/rt}
here="$(cd "$(dirname "$0")/.." && pwd)"
CHUNK=2
cd "$here/rt"
mkdir -p "$here/$out" "$here/cache/live"
per=$(( (count + workers - 1) / workers ))
for w in $(seq 0 $((workers - 1))); do
  start=$(( first + w * per ))
  n=$(( per < first + count - start ? per : first + count - start ))
  [ "$n" -gt 0 ] || continue
  # A fresh process every few worlds: the ray tracer keeps compiled kernels and scene buffers of every
  # world it has seen, so one long process grows until the machine runs out of memory.
  (
    for chunk in $(seq "$start" "$CHUNK" $((start + n - 1))); do
      size=$(( CHUNK < start + n - chunk ? CHUNK : start + n - chunk ))
      nice -n 10 ../.venv-rt/bin/python generate.py --out "$here/$out" --first-room "$chunk" --rooms "$size" \
        --fast --live "$here/cache/live/worker$w.json"
    done
  ) >> "$here/$out/generate_$w.log" 2>&1 &
done
echo "running; live view: (cd $here && python -m http.server 8765) then open http://127.0.0.1:8765/rt/live.html"
wait
