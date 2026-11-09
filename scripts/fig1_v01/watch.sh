#!/bin/bash
# Exit with a message on the first milestone or failure (used to wake the session).
B=runs/fig1-v01/block; SP=runs/stem-trial-v01/pac
seen_pac=$([ -s $SP/D0/fit.json ] && echo 1 || echo 0)
while true; do
  f=$(systemctl --user list-units 'fig1-*' 'blk-*' 'stem-*' --all --no-legend 2>/dev/null | awk '$3=="failed"{print $1}')
  [ -n "$f" ] && { echo "FAILED: $f"; exit 1; }
  if [ $seen_pac = 0 ] && [ -s $SP/D0/fit.json ]; then echo "MILESTONE: PAC-NeRF stem D0 fit done: $(cat $SP/D0/fit.json | tr -d '\n' | cut -c1-200)"; exit 0; fi
  n=$(ls $B/gic/forecast-D2-*.npz $B/pac/forecast-D2-*.npz 2>/dev/null | wc -l)
  [ $n -ge 12 ] && { echo "MILESTONE: all S2 forecasts present"; exit 0; }
  sleep 120
done
