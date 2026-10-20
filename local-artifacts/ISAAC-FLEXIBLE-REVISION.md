# Revised flexible-block recordings

The user found the original point-pin video too rigid. This revision uses1kPa material,20 fixed support nodes and three9-node dynamic load patches. The matched20kPa control uses exactly the same forces/boundaries/camera. Elastic coordinate RMS rises0.782→5.745mm (7.35x); no displacement amplification. Minimum soft tetrahedron volume ratio0.274. Point-load inversion was rejected with failure state and exit1. Numerical/spatial convergence remains open.

All215 raw files, including every rejected trial, RGB pair, source snapshots and logs, were roundtrip hash checked. Reassemble `cat isaac-flexible-revision.tar.xz.part-* > isaac-flexible-revision.tar.xz`, verify SHA256 71b631a6f4b17d75d0a8b228b4442c54ff3fa93b66ec8d0af96fbcccd8a3cb2d, then extract with `tar -xJf`. Every uploaded part matches its local Git blob hash. Videos and updated conversion helper are on main under examples/isaac-flexible-block.
