#!/bin/bash
# Relaunch the remaining Figure-1 GPU work after a reboot. Every step is idempotent (skipped when its output exists).
# History: the card at PCI 0000:67:00 (UUID GPU-571378aa..., CUDA index 2) "fell off the bus" (Xid 79) under load on
# 2026-11-10 00:16, 2026-11-10 19:48 and 2026-11-11 11:57 (water-cooling problem), each time forcing a node reboot.
# It is EXCLUDED: jobs are pinned to the three healthy cards by UUID (immune to index reordering). Its former share
# (PAC block Bcentre, M-PhyGs block Bleft) is appended to card A's queues. One PAC + one M-PhyGs per card fits in
# 11 GB (never two M-PhyGs per card).
set -eu
R=/home/bayespc2/vscode/guassianSDE; U=$R/scripts/stem_trial_v01/run_unit.sh; Q=$R/scripts/fig1_v01/queue_gpu.sh
P=$R/scripts/fig1_v01/mphygs/pipeline.sh; L=$R/runs/fig1-v01/logs; ML=$R/runs/fig1-v01/mphygs/logs
A=GPU-5b432ce5-a24a-68ba-150f-682265260a9b   # PCI 19:00
B=GPU-c49707f6-24fd-38e4-3973-3543ec644ceb   # PCI 1A:00 (also showed hardware thermal slowdown)
D=GPU-9dbe9939-379d-9560-2028-462a2f8b3ac1   # PCI 68:00 (drives the display)
if nvidia-smi 2>&1 | grep -q 'Unknown Error'; then echo "driver reports a lost GPU -- reboot first"; exit 1; fi
for g in $A $B $D; do nvidia-smi -L | grep -q $g || { echo "healthy card $g missing"; exit 1; }
  CUDA_VISIBLE_DEVICES=$g $R/runs/gap-crossmethod-v02/mphygs-env/bin/python -c \
  "import torch;x=torch.randn(2048,2048,device='cuda');print('$g ok',float((x@x).sum())>-1e30)" || { echo "card $g failed smoke test"; exit 1; }; done
mkdir -p $R/runs/gpu-health
systemctl --user is-active -q gpu-guard || $U gpu-guard 512M $R $R/runs/gpu-health/guard.log -- /usr/bin/python3 $R/scripts/fig1_v01/gpu_guard.py
systemctl --user is-active -q fig1-timemon || $U fig1-timemon 1G $R $L/time-monitor.log -- /usr/bin/python3 $R/scripts/fig1_v01/time_monitor.py
$U fig1-qA 32G $R $L/queue-gpuA.log -- /bin/bash -c "$Q $A - W500; $Q $A Bcentre -"
$U fig1-qB 32G $R $L/queue-gpuB.log -- $Q $B Bleft W1000
$U fig1-qD 32G $R $L/queue-gpuD.log -- $Q $D Bright W2000
$U mphygs-gpuA 12G $R $ML/mphygs-gpuA.log CUDA_VISIBLE_DEVICES=$A -- /bin/bash -c "$P stem W500 && $P block Bleft"
$U mphygs-gpuB 12G $R $ML/mphygs-gpuB.log CUDA_VISIBLE_DEVICES=$B -- /bin/bash -c "$P block Bcentre && $P stem W1000"
$U mphygs-gpuD 12G $R $ML/mphygs-gpuD.log CUDA_VISIBLE_DEVICES=$D -- /bin/bash -c "$P block Bright && $P stem W2000"
echo "relaunched on 3 cards; watch with: $R/scripts/fig1_v01/watch.sh"
