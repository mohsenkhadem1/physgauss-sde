# Newton-law P2 axial spatial reference

The declared static case passes all prospective C38 gates. This uses Newton 1.7's stable Neo-Hookean continuum energy, with the standard E/nu conversion and kappa=lambda+mu, rather than the earlier log-J material law.

The block is 0.24 x 0.08 x 0.08 m, E=5000 Pa, nu=0.35 and rho=625 kg/m3. A 1 N uniform right-face axial dead traction acts against the exact full left-face clamp; gravity is zero. Both 12,000-tetrahedron P2 equilibria (18,081 nodes, quadrature orders 3 and 4) started independently from rest.

| Check | Result | Declared criterion |
|---|---:|---:|
| 6144-to-12000 maximum displacement difference at the fixed 122 probes | 3.641916 um | <=20 um |
| Conditional p>=1 fine-error indicator, 4 x that difference | 14.567663 um | <=20 um |
| Q3-to-Q4 maximum probe difference | 0.000030664 um | <=20 um |
| Original free equilibrium residual, Q4 | 5.191e-12 N | <=1e-11 N |
| Force / actual moment balance defects, Q4 | 2.96e-13 N / 8.74e-13 Nm | <=1e-8 each |
| Clamp error | 0 m | exactly zero |
| Sampled minimum determinant J, Q4 | 0.989739 | >0.1 |
| Sufficient global J lower bound, Q4 | 0.757350 | >0 |
| Sampled maximum principal stretch strain, Q4 | 6.4542% | 2-8% |
| Volume nonrigid coordinate RMS, Q4 | 1.293318 mm | >=1 mm |

The global determinant certificate follows from the affine P2 displacement gradient on each straight tetrahedron: the largest vertex spectral norm of grad(u) is 0.088482<1. The positive lower bound applies throughout each element. Sampled peak strain, volume-weighted 99th percentile (3.3235%) and volume RMS strain (2.0311%) are reported separately; no clamp-edge points are discarded.

The 4 x difference is a conditional first-order engineering indicator, not a rigorous continuum error bound. This qualifies the declared static case at these fixed displacement queries. It does not qualify dynamics, other loads/supports, gravity/contact, neural training or inference. The previous stable-law 768-to-6144 query failure (22.822907 um) remains recorded; the old log-J and native P1 results are unchanged.

Both solves completed in 542.23 s using two CPU threads. The independent watchdog observed peak resident memory 2,744,344,576 bytes, below its frozen 8 GiB limit; the frozen 900 s cap was not reached. Worker and watchdog are absent and the exec session was reaped with exit 0. All source and capture hashes were reverified after completion.

Sources and evidence: [protocol](../../operator-dev/newton-stable-p2-spatial-v01/protocol.json), [summary](summary.json), [terminal receipt](../../operator-dev/newton-stable-p2-spatial-v01/terminal.json), [resource receipt](resource-guard.json). Raw fields remain local in the two hashed capture files.
