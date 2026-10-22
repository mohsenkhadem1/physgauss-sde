# Native rectangular-block gravity and drop demonstrations

All eight declared cases completed in 162.42 s on GPU0, one Isaac app at a time. The block is 0.24×0.08×0.08 m, 325 nodes/1152 tetrahedra, nominal mass 0.96 kg. Five gravity cases use E=300 kPa with left/right/both-end/bottom/two-finite-bottom-patch supports; the free-fall control uses the same material; two floor drops use E=300/600 kPa. Density625 kg/m³, nu=.35, gravity9.81 m/s², PGS32 and240 Hz are explicit. No handles or force attachments are present.

Native E/nu, rest tetrahedra, direct nodal target enabled/free flags, zero initial velocity, gravity readback and actual float32 step clock were verified. All fixed-node drift was zero. Density is authored; no native nodal-mass readback is claimed. The exposed installed API did not identify the constitutive energy, so this is not asserted to share the CPU Neo-Hookean law.

The free-fall control passed its predeclared velocity/nonuniformity/J checks: peak COM-velocity error1.71 µm/s, maximum|J−1|2.30e-6, translation-removed coordinate RMS4.56 nm. Position agrees with kick-drift to9.53 nm coordinate RMS; exact held-acceleration position differs by1.384 mm. Both references are retained; no fitted integration convention certifies material physics.

**Timestep accuracy fails.** Three fresh identical-mesh left-clamp gravity captures at240/480/960 Hz completed in52.00 s. Maximum same-material-node differences were15.514 mm,5.645 mm and21.157 mm for240→480,480→960 and240→960, respectively, against the fixed20 µm threshold. No continuum, spatial, constitutive or contact accuracy follows from the demos. The earlier temporalv01 directory-setup failure happened before any Isaac/GPU call and remains preserved; v02 adds the missing output parent and its test.

A separately frozen observation-only recapture saved all145 actual240 Hz states of both drops in31.83 s, without changing physics. Maximum compatible-P1 principal-stretch departure was14.025%/13.830%; peak volume-weighted proper-rigid-removed coordinate RMS was0.652/0.411 mm. These are local strain and body-mean coordinate metrics, respectively. Dense states reveal3.529/2.826 mm peak nodal floor penetration; sparse30 fps snapshots missed the brief peak. No interpolation, displacement exaggeration, exact nonpenetration or contact-accuracy claim is appropriate.

All controller and worker PIDs are absent; GPU0 is released. NPZ trajectories and videos stay local. Source, frozen protocols, tests and compact result metadata are publishable.
