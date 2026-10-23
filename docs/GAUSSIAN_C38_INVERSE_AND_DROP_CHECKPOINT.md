# C38 inverse test and drop convergence checkpoint

The static mechanics reference is ready for controlled unknown-physics tests.
The drop reference remains unsuitable for a prediction dataset.

Forty static states passed the prescribed mechanics and two-mesh checks. These
cover two support patterns, five stiffness values and four known force states.
The largest conditional 32-to-40-cell displacement indicator at the 63 declared
material points is 13.381 micrometres, below the 20 micrometre gate. Interpolation
from three stiffness nodes predicts the two withheld stiffness values with a
maximum point error of 0.01987 micrometres across the declared validation slices.
Those are finite checks, not a continuous-material or full-field certificate.

A linear elasticity baseline supplied with the true stiffness and support has
3.7884 micrometres pooled coordinate RMSE on the finest static reference fields,
versus 686.595 micrometres for leaving the object at rest. This is an oracle
forward-model result. It does not establish unknown-parameter inference or
justify a neural network. Removing the best rigid motion leaves 46–60% of bulk
motion RMS in these fourteen fields; the experiments contain real deformation.

The next test hides stiffness and support. It uses four fixed physical
configurations, sixteen independent sensor-noise repetitions each, nine ideal
tracked points and stipulated 200 micrometre Cartesian noise. The 64 records are
not 64 independent population objects, RGB-D measurements or temporal forecasts.
Nonlinear-table and linear-model predictions are computed from gravity-only
observations. Every prediction and every planned poke choice must be saved and
hash-bound before a separate evaluator releases labels or canonical targets.
The action comparison includes fixed down, fixed up and random choices at the
same one-probe observation budget. No calibration or active-policy advantage is
claimed while those results are pending.

Posterior controls use adaptive integration and independent 64/128-point
Gauss–Legendre integration on explicit polynomial root partitions. A preserved
counterexample demonstrates why agreement between generic integrations alone
can miss a narrow posterior peak. Model interval coverage and numerical quantile
root enclosures are reported separately. No sensor-noise floor is added to the
noiseless predicted deformation field.

The drop uses a finite compliant frictionless floor, consistent quadratic-element
mass and an explicit fixed contact operator. Time refinements are encouraging:
the 96-element mesh differs by 7.727 nanometres and 32.384 micrometres/second;
the 324-element mesh differs by 5.025 nanometres and 31.151 micrometres/second.
The 768-element mesh passes the position time gate at 17.326 nanometres but
fails the velocity gate at 192.421 micrometres/second (limit 100).

More seriously, comparison at the same 81 material points and nine physical
times gives raw adjacent-mesh position differences of 2.50050 and 1.56223 mm,
and velocity differences of 3.42379 and 2.37502 m/s. Raw errors contract, but the
finest conditional indicators remain 4.68668 mm and 7.12506 m/s, far above the
20 micrometre and 0.1 mm/s gates. Temporal and spatial checks remain separate.
These trajectories are development diagnostics, not admitted ground truth.
Further work must explain the spatial discrepancy rather than treat conservation
or accurate time integration as proof of a converged simulation.

Raw arrays, private mappings, videos and model files remain local. Source,
protocols, negative results and compact completed reports are published.
