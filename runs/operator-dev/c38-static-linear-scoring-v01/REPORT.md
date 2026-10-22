# Fixed supplied-physics linear comparison

All 28 case/resolution comparisons completed in 27.651 s. Predictions were sealed before nonlinear target array reads; no parameters, model, noise or roles were fitted/selected. The comparison uses the same known material and support (oracleA), not image-based or sparse-view inference. All 40 source pins and capture hashes remain unchanged; session61253 was reaped with exit0 and its PID is absent.

At the finest mesh, pooled coordinate RMSE over the same63 reference points and14 engineering configurations is **3.7884 µm** for the linear model, **686.595 µm** for rest/no-motion and **438.739 µm** for the rigid motion fitted to linear predictions. The largest individual linear point error is **23.2743 µm**. This is prediction approximation error, separate from either model’s own20 µm mesh-indicator gate; no universal20 µm prediction-accuracy claim is made.

The rigid fit to nonlinear targets minimizes the positive-Keast bulk volume objective only. Its evaluated q63 RMSE (438.853 µm) is descriptive and is **not** a q63-best-rigid lower bound or deployable predictor. Co-rotated deformation errors and bulk nonrigid magnitudes are likewise descriptive saved-field diagnostics, with no whole-field spatial accuracy certificate.

| Finest configuration | Linear q63 coordinate RMSE (µm) | Largest point error (µm) | Rest q63 RMSE (mm) |
|---|---:|---:|---:|
|interior-gravity__both_ends|2.0504|5.5543|0.8102|
|interior-gravity__separated_bottom_patches|2.1408|5.4914|0.2709|
|Elo__both_ends__down2|7.6652|21.7109|1.5111|
|Elo__both_ends__gravity|4.1084|11.1711|1.1456|
|Elo__both_ends__up2|2.4329|7.0882|0.7825|
|Elo__separated_bottom_patches__down2|8.8343|23.2743|0.5327|
|Elo__separated_bottom_patches__gravity|4.3422|11.1585|0.3843|
|Elo__separated_bottom_patches__up2|1.6171|4.0447|0.2406|
|Ehi__both_ends__down2|1.9510|5.6252|0.7551|
|Ehi__both_ends__gravity|1.0242|2.7661|0.5729|
|Ehi__both_ends__up2|0.6006|1.7377|0.3916|
|Ehi__separated_bottom_patches__down2|2.1388|5.6103|0.2644|
|Ehi__separated_bottom_patches__gravity|1.0601|2.7160|0.1912|
|Ehi__separated_bottom_patches__up2|0.3980|0.9925|0.1200|

All28 linear fields have positive-J flags, and all own32→40 q63 linear spatial indicators pass (1.67455–11.91788 µm). The nonlinear q63 mesh qualification is separate. Current mechanics do not qualify arbitrary points, dense pixels or peak stress; all historical failures and the original metadata-loss statement remain in the forward report.

The physical linear baseline is already very accurate for these supplied-physics engineering cases. This does not establish prediction with unknown E/support, posterior calibration, adaptive action benefit, or a reason to add a neural network. It is not a temporal forecast, population test or image pipeline. Raw fields remain LOCAL.

[Aggregate](aggregate.json) · [Terminal proof](terminal-proof.json) · [All28 detailed q63/diagnostic metrics](../../gaussian-campaign-v01/c38-static-linear-scoring-v01/summary.json).
