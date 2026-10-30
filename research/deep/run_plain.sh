#!/bin/sh
for enc in cnn2d cnn1d transformer spectral; do
  .venv/bin/python -m deep.train --name ${enc}_plain --encoder $enc 2>&1 | grep -v Warning
done
