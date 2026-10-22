# Fixed96tet Q8 backend bridge passes

The newWarp trajectory and completed tightTorch reference use identical96tet P2 geometry, fullconsistentM, stablematerial, finite-compliantQ8 surfacecontact, gravity, analyticinitializer,1e-10 numericalsettings and fixed9-outputclock. Their all-node positiondifference **1.202nm** passes10nm and velocitydifference **3.396µm/s** passes10µm/s. Output-energy difference0.566nJ passes470.88nJ. This isolates the backend before any surfacequadrature change.

Warp completed25,613 acceptedsteps/707,411RHS evaluations. Acceptedenergyerror2.680nJ, momentum1.02e-14Ns, masssolve2.07e-13N and all declared positive-J guards pass. Integration888.67s, wholeworker891.87s, controller894.95s; no timeout/stop signal. The previousTorch reference is imported with source/capture hashes and is not rerun.

The correctedresource guards passed atstartup, every512steps and beforeprojectedcheckpoint/finalwrites: ownGPU8GiB, RSS12GiB, inheritedCPU2 affinity/threads, and filesystemfree minusprojectedwrite≥30GiB. The compact sampled envelope accompanies this result. Fullcommitted checkpoints include y/f/next_h_abs; rawstates/logs/cache staylocal. Controller/worker are reaped; GPU1released.

The two earlier bridgeprotocol revisions were superseded before execution and remain unchanged. This completed bridge supports the stated backend on this case; it does **not** qualify surfaceQ, meshresolution, continuumcontact, rigidnonpenetration or droptrainingdata. The next comparison must use the sameWarp backend atQ8 andQ16 under a separately reviewed protocol.
