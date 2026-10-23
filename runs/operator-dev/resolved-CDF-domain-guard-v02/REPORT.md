# Numerical CDF central-estimate guard repair

The first resolved-quantile matrix retains numerical negatives, including a
linear prefix whose CDF trace gives central estimate `1.0000000000000002`, lower
bound `0.9999999999999336`, upper bound `1.0`. Only the test that the central
estimate lies below its range-intersected upper bound fails. This trace uses
public inputs and never private labels/targets or another action's observations.

The separate v02 helper preserves the supplied numerical CDF bounds exactly.
When an estimate lies outside the mathematical CDF range [0,1], and the provided
enclosure touches the corresponding endpoint, only that central estimate is
projected onto [0,1]. Root signs and returned probability guarantees continue to
use the original bounds. An inconsistent interior estimate, nonfinite value or
invalid bounds still fails. No positional tolerance, probability tolerance,
likelihood, prior, physical prediction or interval margin changes.

Five controls pass, including the exact observed tuple, both boundary directions,
rejection of invalid bounds/interior contradictions, and identical values,
brackets and query counts on inputs accepted by v01. Successful v01 results can
be referenced by hash under this compatibility argument; any failed model arm
requires a separately recorded v02 execution. Original v01 negatives remain
preserved, and this source alone does not claim that every later failure has
the same cause. No actual model arm has been rerun by this helper's controls.
