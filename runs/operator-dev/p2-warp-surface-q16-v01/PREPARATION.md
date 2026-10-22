# Same-backend surfaceQ16 preparation

This is a source/preparation snapshot, not a Q16 trajectory result. The completed96tet Q8 Warp capture is imported once; neither oldTorch nor Q8 is rerun. The newcase changes onlysurface integration fromQ8 (64points/triangle) toQ16 (256points/triangle). It preserves Warp force evaluation, stablematerial, bulkquadrature/guards, fullconsistentM, kappa1e9 finite-compliantplane, initializer/clock,1e-10 accuracy and maximumstep1/30000s.

Prospective comparison gates are all-node and81fixed-query positions20µm, all-node velocity0.1mm/s, plus unchanged acceptedenergy/P/J/mass checks. Guarded3600s GPU1/CPU2 budget retains8GiB ownGPU,12GiB RSS, freefilesystem minusprojectedwrite≥30GiB and ownedTERM/KILL/reap cleanup. Fullcommitted y/f/next-h checkpoints remain. Fourtimes surfacepoints/uncertainactivationcount motivate the budget, not a runtimeguarantee. Failure/cap is retained; noautomaticQ32 followup.

The separate CPU-only control verifies actualQ16 dimensions, rest/inactivecontact and synthetic P2partialcontact against the independentTorchCPU evaluator. It passed strictforce/energy/J thresholds without reading any actualtrajectory values or usingGPU. The reviewed source/protocol remained unchanged. Surface/mesh/rigidfloor qualification is not implied by preparation or this manufactured control.
