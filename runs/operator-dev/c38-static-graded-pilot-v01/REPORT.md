# Graded two-case pilot: terminal displacement checks

Both cases completed in 179.45 s. All six mesh solves and two independent coarse CPU/GPU controls passed the physical checks. **Both cases still fail the unchanged 20 µm spatial indicator; no family or dataset is admitted.**

| Case |Uniform 20→24 (µm)|Graded 20→24 (µm)|Uniform peak strain (%)|Graded peak strain (%)|Graded bulk RMS (mm)|
|---|---:|---:|---:|---:|---:|
|Elo__separated_bottom_patches__down2|73.838|33.348|9.706|31.572|0.3322|
|Elo__both_ends__down2|24.436|23.997|7.660|12.079|0.7808|

Grading reduces the bottom-supported displacement indicator substantially, but its sampled peak strain rises to 31.57% near fixed support/patch edges. The end-supported control reaches 12.08%. These peaks are not converged stress measures and do not establish moderate local strain or real-material validity. The original polynomial stable-NH law, full forces, supports, all 63 probes and gates are unchanged.

Next planned numerical test: preserve and hash-bind each completed graded 24 field, then start new graded 32 and 40 solves from rest on these same two cases. No 24 rerun, extra case or physical retuning. This can test conditional displacement convergence only; passing two cases would not qualify all 14 original cases. Prospective source/protocol review precedes any new launch.

Raw captures and accepted checkpoints stay local. Controller session 93966 reaped exit 0; controller and both worker PIDs are absent, all runtime source pins and six capture hashes verified. Exit 0 denotes completion, not scientific success.
