#!/bin/bash
# queue.sh JOBFILE : run Fig1Comsol jobs sequentially; each line "OUTNAME SCEN WORLD HMAX RTOL MAXSTEP [HTIP]"
O=/home/bayespc2/vscode/guassianSDE/runs/fig1-v01/comsol
while read -r name scen world h r dt tip; do
  [ -z "$name" ] && continue; [[ "$name" == \#* ]] && continue
  [ -f "$O/$name.json" ] && { echo "skip $name"; continue; }
  echo "=== $(date) start $name"
  /home/bayespc2/vscode/guassianSDE/scripts/fig1_v01/comsol/run.sh $scen $world $h $r $dt $name $tip > "$O/$name.log" 2>&1
  echo "=== $(date) end $name rc=$?"
done < "$1"
