Renderer numerical qualification — development evidence

renderer-numerical-discrepancy.png/.svg show actual unmodified factor1 RGB and optical depth for development-3, opposed translation, left camera, using the same physical parameters at n4/n6. Only difference panels are amplified: RGB difference ×100 and enlarged magenta markers at eight changed silhouette pixels. No geometry deformation is exaggerated.

For that case, feature Mahalanobis squared distance is 783.1193 / .002611 / .000664 at factors1/2/4. Across all20 prespecified pairs, maximum distances are783.1193 /4.6715 /7.6955. Better n4/n6 agreement does not establish image convergence: fixed-geometry2→4 differences have median126.5173 and maximum1006.9667 under the unchanged nominal covariance.

Decision: stop raster refinement under the bounded development plan. Retain original factor1 renderer and all original banks. Subsequent finite-ensemble policy evaluation is exploratory and has demonstrated numerical observation mismatch; it must not claim a converged continuous posterior or numerically qualified sensor likelihood. No sensor-noise inflation or reserved data access was used.

Supersampling integrates existing RGB and far-clipped depth signals within pixel footprints. It is a diagnostic approximation, not a calibrated model of mixed depth returns. Eighty renders took22.4s on four workers; factor4 was not adopted.

Source evidence: ../render-discrepancy.json, ../supersampling-diagnostic-plan.json, ../supersampling-diagnostic.json, ../SUPERSAMPLING-QUALIFICATION.txt. Figure input/output SHA256 hashes are in renderer-numerical-discrepancy-provenance.json. Rebuild from repository root with runs/phystwin-env/bin/python reports/physical-refinement-v15/figures/build_renderer_diagnostic.py. Figure layout was visually inspected; no production source or bank was changed.
