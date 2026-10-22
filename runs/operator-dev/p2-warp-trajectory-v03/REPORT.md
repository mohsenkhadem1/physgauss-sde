# Tighter paired trajectory control passes

FreshTorch andWarp6tet trajectories with identical1e-10 relative,1e-13m displacement and1e-11m/s velocity tolerances agree to **7.452nm** at all common nodal positions and **7.852µm/s** at all common nodal velocities. These pass the unchanged10nm and10µm/s implementation gates. The earlier1e-9 negative comparison remains preserved; its gates were not relaxed.

Both use exactly the same stablematerial, fullconsistentM/Cholesky, gravity, quadraticQ8 finite-compliance contact, analytic precontact initializer,0.15s endpoint and9 common120Hz outputs. Allaccepted conservation checks pass: Torchenergy9.920nJ, Warp4.993nJ, both below470.88nJ; momentum below7.57e-15Ns and masssolve below4.02e-13N. Both preserve positive stage Jacobians. Fullcommitted checkpoints include y/f/next_h_abs.

Torch completed7157 acceptedsteps/182039RHS in391.665s andWarp7163/180083 in156.021s; these are actual per-case times, not equal-RHS timing. The separate paired same-state benchmark already measured2.265× inclusive evaluation speed. Wholephase551.55s; controller/worker reaped andGPU1released.

This supports the new force backend for the stated implementation control. It is not temporal/spatial/surface-quadrature qualification or hardnonpenetration. Rootreviewed surface-Q refinement is still needed on the96tet model. Source/packages/oldresults remain unchanged; rawNPZ/checkpoints/logs staylocal.
