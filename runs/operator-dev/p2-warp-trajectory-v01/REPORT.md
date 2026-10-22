# Paired6tet backend control:300s resource limit

The originalTorch case completed in275.689s and passed its unchanged conservation/material guards: energy87.142nJ below470.88nJ, momentum6.46e-15Ns, minimumJ0.928035, masssolve3.46e-13N. Its5360 acceptedsteps/130199 force evaluations and all9 outputs are retained locally with hashes.

The freshWarp case reached its last committed1024 acceptedsteps at0.10210266s before the owned300s limit. Both controller/worker are reaped; the phase is incomplete and does not qualify a trajectory backend. There was no recorded physical/material/force failure. The partial committedprefix is retained. A new explicitly bounded phase can import the completedTorch result and execute onlyWarp; no oldbudget or source is changed.
