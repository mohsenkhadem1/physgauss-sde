# Guard-only v02 continuation: partial repair, two negatives retained

The complete resolved v01 run is immutable: 254 numerical passes and two failed linear-prefix arms, with all 256 posterior cores matching the original reference. The v02 continuation references all 254 successful arms by exact file hashes. It recomputed only the two failed model arms from the same public observations, models and prior. No successful arm, physical observation, policy or noise draw was repeated.

The sole numerical change is the reviewed central-CDF domain projection when a supplied bound touches 0 or 1. Supplied bounds, root branches, positional/probability tolerances, model, prior and posterior moments are unchanged. The one-case public trace established this cause for the original adaptive failure only.

Both adaptive computations now finish. Both GL128 target-quantile computations still raise `Finite ordered numerical CDF enclosure required`. Consequently the merged record contains **254 numerical passes and two retained negatives**, not an all-pass result. The fix has not been broadened and no additional retry occurred. All 256 normalizer/category/material/moment core comparisons remain bit-identical. Calculated quantiles are not posthoc uncertainty margins, and no physical or population calibration claim follows.

The continuation executed only two single-model jobs, exited 0 after 27.146 seconds and was reaped. All three new and 129 original owned PIDs are absent. Every merged model file/coreproof/source-result/protocol hash is verified; origins and original failure denominators are explicit in the merged seal. Raw posterior arrays remain local. No private canonical-target or parameter-label analysis occurred in this controller.
