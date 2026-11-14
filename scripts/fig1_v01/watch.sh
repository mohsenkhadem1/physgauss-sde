#!/bin/bash
# Exit with a message on the first milestone or failure (used to wake the session).
B=runs/fig1-v01/block; SP=runs/stem-trial-v01/pac; L=runs/fig1-v01/logs
tb() { cat $L/queue-gpu*.log runs/fig1-v01/mphygs/logs/*.log runs/fig1-v01/phystwin/logs/*.log 2>/dev/null | grep -c Traceback; }
seen_pac=$([ -s $SP/D0/fit.json ] && echo 1 || echo 0); tb0=$(tb)
while true; do
  f=$(systemctl --user list-units 'fig1-*' 'blk-*' 'stem-*' 'tcnn-*' 'mphygs-*' 'ptw-*' --all --no-legend 2>/dev/null | awk '$3=="failed"{print $1}')
  [ -n "$f" ] && { echo "FAILED: $f"; exit 1; }
  [ "$(tb)" -gt "$tb0" ] && { echo "FAILED: new Traceback in a queue log"; grep -l Traceback $L/queue-gpu*.log; exit 1; }
  if [ $seen_pac = 0 ] && [ -s $SP/D0/fit.json ]; then echo "MILESTONE: PAC-NeRF stem D0 fit done"; exit 0; fi
  a=$(systemctl --user list-units 'fig1-q*' 'mphygs-gpu*' --state=active --no-legend 2>/dev/null | wc -l)
  [ "$a" -eq 0 ] && { echo "MILESTONE: all Figure-1 queues and M-PhyGs runs finished"; exit 0; }
  sleep 120
done
