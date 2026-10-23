# Drop failure is dominated by unresolved nonrigid motion

The CPU-only read-only diagnosis completed in61.25 s. No mechanics trajectory was rerun and no old source, force law or gate changed. It assessed all27 predeclared saved output states from the96/324/768-tet root128 tight captures.

**The velocity failure is predominantly internal deformation, rather than a different overall bounce.** For324→768, the common-query velocity discrepancy has coordinate RMS0.40237 m/s. Its fitted rigid component is0.00147 m/s, only0.365% of its L2 norm, while the nonrigid component is0.40222 m/s. The maximum nonrigid query-velocity difference remains2.3742 m/s, almost the entire raw2.3750 m/s. The corresponding position RMS is175.54 µm: rigid33.08 µm and nonrigid175.55 µm, with a negative cross term; these are not additive percentages. Removing rigid motion therefore does not rescue the old gates.

The finest meshes' maximum COM differences are88.59 µm and5.529 mm/s. These still exceed the strict targets, but are much smaller than the local query differences1.562 mm/2.375 m/s. Positive reference-volume quadrature gives peak bulk nonrigid velocity RMS0.347/0.355/0.357 m/s across the three meshes, despite large differences between corresponding local velocity fields. This supports spatially unresolved elastic-wave/phase behaviour. It is not a unique modal identification or a proof of causality, especially with only nine physical output clocks.

**No sampled contact-force quadrature mismatch was found.** Across all27 states, CPUroot128 agrees with the independent NumPy/strict adaptive E/g reference to1.46e-13 N global gradient norm and2.78e-17 J energy. CPUroot64 versus128 differs by at most7.66e-14 N. Every adaptive reference completed; no status waiver or partial sum was used. This fixed-state evidence cannot bound the effect of unsampled trajectory states, and does not rehabilitate the old strict Hessian-reference failures.

Exact consistent-mass momentum agrees with the saved integrated net-force auxiliary state within6.7e-15 Ns. The first accepted cumulative contact impulse exceeding the prospectively fixed1e-8 Ns threshold occurs at0.100963918/0.100963920/0.100963798 s; these sampling proxies differ by about0.12 µs and are not exact event-time certificates. Final upward contact impulses are1.67689/1.69406/1.69250 Ns. COM/momentum therefore show neither a force-magnitude error nor a large global contact-onset shift capable of explaining the local velocity mismatch by themselves.

Position Kabsch fits use positive Keast24 reference-density quadrature, exact for the relevant polynomial products. Instantaneous velocity fits solve the weighted rigid-velocity least-squares problem on the current configuration, rather than differentiating the fitted rotation. Consistent P2 nodal integral weights are used only for exact linear COM/P integrals; their possible negative coefficients are not treated as negative physical point masses. Quadrature versus consistent-mass COM and COM-velocity agree within1.3e-15 m and4.2e-14 m/s.

The original root128 spatial failures and768 temporal velocity failure remain unchanged. The proposed1e-11 temporal run was not launched and is on HOLD: better time accuracy alone cannot close millimetre/metre-per-second spatial errors.

## Separately authorised new physical experiment

Root selected a **different compliant boundary**, κ=1e6 N/m³ instead of1e9, keeping the5 cm drop, E300 kPa, nu.35, rho625, gravity and all numerical gates fixed. It must be labelled a softer-floor experiment, not a repair or reclassification of the old impact.

A simple scale calculation explains its purpose, not its outcome. With contact area0.0192 m², the floor stiffness scale falls from19.2 MN/m to19.2 kN/m. A uniform axial body stiffness scale E*A/H is72 kN/m. A series-spring estimate therefore changes from71.73 to15.16 kN/m, reducing the approximate global bounce frequency from43.5 to20.0 Hz. Floor-dominated local stiffness frequencies scale by sqrt(1000), while real elastic modes need not follow that simple estimate. Idealised energy partition suggests roughly2% body strain could remain, but this is not simulated evidence or a guarantee of visible/resolved motion.

The new physical screen will retain every outcome and must satisfy fresh time/surface/space evidence before any contact dataset admission. Lowering drop height alone would reduce impact energy/amplitude, not the linear natural frequencies; it is not being substituted silently. No noise increase, velocity filtering, rigid projection or damping is proposed.
