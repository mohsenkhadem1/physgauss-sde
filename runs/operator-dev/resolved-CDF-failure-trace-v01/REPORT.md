# Raw CDF guard failure: one public posterior trace

The immutable v01 failure was reproduced using only the same acquired public linear-prefix likelihood, adaptive posterior and hypothetical target polynomial. After two CDF queries the raw tuple was `cdf=1.0000000000000002`, `lower=0.9999999999999336`, `upper=1.0`. All entries were finite; the supplied bounds were ordered and inside [0,1]. The sole failed guard condition was central estimate ≤ upper bound.

The diagnostic returned the raw dictionary unchanged and made zero v02 guard calls. It changed no production source, likelihood, observations, prior, root tolerance or intervals. This identifies a one-ULP central-estimate overshoot for this specific failure; it does not establish that every failed arm has the same cause or that a corrected posterior is calibrated.

The CDF trace took 1.482 seconds; process startup, source/input preflight and tracing took 9.282 seconds. The owned process exited 0 and was reaped, both owned PIDs are absent and all pinned files remain unchanged. Six manufactured source controls passed. Detailed query values and point identity remain local; only the offending tuple, condition flags, counts and provenance are published.
