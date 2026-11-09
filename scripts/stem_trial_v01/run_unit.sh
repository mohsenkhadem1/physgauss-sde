#!/bin/bash
# Run a job as its own systemd user service with a hard memory cap, so a runaway job is
# killed inside its own cgroup instead of driving session-wide memory pressure (systemd-oomd
# then kills the whole VS Code scope, which happened twice on 2026-11-08).
# Usage: run_unit.sh NAME MEMMAX WORKDIR LOG [ENV=VAL ...] -- CMD ...
name=$1; mem=$2; wd=$3; log=$4; shift 4; envs=()
while [ "$1" != "--" ]; do envs+=(-E "$1"); shift; done; shift
systemctl --user reset-failed "$name" 2>/dev/null
systemd-run --user --unit="$name" --working-directory="$wd" -p MemoryMax="$mem" -p MemorySwapMax=0 \
  -p StandardOutput=append:"$log" -p StandardError=append:"$log" "${envs[@]}" "$@"
