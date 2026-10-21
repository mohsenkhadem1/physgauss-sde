# Authored-force command compiler

The new compiler preserves each exact authored peak and original `forces_at` float32 output, promoted to float64 for the independent solver. All four public TRAIN fixtures originally use push pulses; pull, shear and couple cases are explicitly labelled interventions on those unchanged layouts. Point-pin peaks already include their original factor of .25, so the compiler applies no further scaling.

Commands use an integer endpoint/midpoint grid. Every command carries an explicit known-actuator or candidate-force role; a recipe value is never automatically a public observation. The audit contains command hashes and closed-form impulse diagnostics. Signed shear impulse is zero, so non-cancelling absolute impulse and the first time moment are also checked. Its nonzero first moment is .24/pi seconds squared for the 3-second unit envelope.

Sixteen tests pass, including exact original-command equality on all four layouts and all four force families, independent dense quadrature of signed/absolute impulses and first moments, force-couple cancellation, and rejection of ambiguous roles or incomplete clocks. These are command tests; no physical trajectories or held-out responses were loaded and no sampler, predictor, or policy was qualified.
