# Paired backend control: strict trajectory agreement fails

The newWarp-only resource phase imported the completed originalTorch trajectory without rerunning it. Both runs used the same6tetP2 model, fullconsistentM, stablematerial/contactlaw, analytic freefall initializer,1e-9 relative/1e-12m displacement/1e-10m/s velocity tolerances and maximumstep1/30000s. All9 common outputs and physical mass/material/clock definitions match.

Warp completed5313 acceptedsteps/129119RHS in111.236s; originalTorch completed5360/130199 in275.689s. Their maximum all-node positiondifference21.9565nm fails10nm and velocitydifference29.0247µm/s fails10µm/s. Both conserve physicalenergy within the unchanged470.88nJ gate (Torch87.142nJ,Warp32.947nJ), and momentum/mass/Jguards pass. This is **not** a qualified trajectory backend.

Strict same-state force/fullRHS parity already passes near2e-13N. Differentadaptivepartitions at1e-9 are a plausible source of the trajectory discrepancy; that explanation requires a controlled tighter-accuracy comparison and does not relabel this failure. No physical parameter, gate or existing trajectory was changed.

The prior300s phase remains incomplete and preserved. ItsWarp checkpoint lacked f/next_h_abs, so the explicitly authorized newphase recomputed the committed21116RHS prefix from the same analytic initializer; no exactresume is claimed. Newcheckpoints now include f/next_h_abs. All rawprefixes and trajectories remain local; controller/worker reaped and GPU1released.
