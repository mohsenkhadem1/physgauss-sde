# Load design, discretization and nonlinear reference checks

This development stage follows the [mechanics priority audit](GAUSSIAN_MECHANICS_PRIORITY.md). It keeps all earlier failed protocols and the sealed prediction comparison intact. Raw trajectories, posterior samples and videos remain local.

## What the diagnosis isolates

The old Gaussian-load simulation is not spatially qualified. The error depends strongly on observation location: between 1,024 and 8,192 tetrahedra, the original corner query differs by about 380 µm, while a fixed 64-point volume probe differs by about 50 µm. These probes are fixed material locations, interpolated from each mesh, rather than mesh-dependent node sets. Removing best-fit rigid motion reduces the interior discrepancy further, but the maximum registered interior error still fails the original noise criterion. Interior averages therefore do not replace or rescue the failed corner/noise gate.

The original refinement preserves the domain and conforming topology, but its element shapes deteriorate and its maximum edge does not halve on each rung. A new shortest-diagonal red refiner bounds the element quality and halves maximum edge length after its first rung. In a matched control with exactly the old material, force, fixture, quadrature and duration, this geometry repair still fails: the final corner RMS difference is **410.19 µm**, against **249.96 µm** tolerance. The finest maximum coordinate-query discrepancy is **1,026.28 µm**. Shape quality is a real issue, but fixing it alone does not resolve the dynamics.

The total lumped mass and centre of mass are already correct. The relative inertia-tensor error nevertheless falls from **23.39% to 1.91%** over the audited old meshes. That can affect rigid rotational response and remains distinct from elastic spatial error.

## Surface integration and load alignment

A surface Gaussian acts through its integrated spatial density, not through its authored centre alone. Independent adaptive integration gives actual rest-surface centroids of approximately **(22.643, 32.659, 19.527) mm** and **(74.007, 70.233, 73.169) mm**. The old opposite-force pair follows the authored-centre axis and therefore has a true rest moment of **9.312 mNm at 1 N**, even with exact integration. It is also compressive. Neither feature is a numerical solver bug; both differ from an intended central tension experiment.

The new surface assembly uses positive triangle quadrature for the normalized load vector ∫ψNᵢ/∫ψ and the full fixture Gram matrix ∫ψNᵢNⱼ/∫ψ. Accurate load integration alone does not make a diagonally lumped fixture consistent. The matrices preserve positive semidefinite elastic and damping energies. Any new dynamic experiment is a separately declared physical/discretization variant, not a retrospective correction of old captures.

## Independent nonlinear mechanical reference

A separate static prismatic-bar benchmark has an independent scalar finite-strain oracle, uniform end traction and a distributed compliant end support. The sparse finite-element solve matches the analytic field on both 48- and 384-tetrahedron meshes at 1 N and 3 N, for both the original area-lumped and fully consistent face-spring assemblies. Maximum position error is approximately **5.85×10⁻¹⁵ m**. The axial strain is **2.12% and 6.58%**; support translation is **5 and 15 mm**.

This verifies the nonlinear constitutive response, compatible boundary assembly and static solver in a homogeneous deformation patch. It does **not** establish localized Gaussian spatial convergence, time accuracy, general boundary inference or a learned forecasting advantage. [Benchmark design and checks](GAUSSIAN_UNIAXIAL_BAR_BENCHMARK.md).

## Revised load screen

The prospective load study tested 1 N rest-central tension with 1.5× broader load standard deviations and 0.4/0.8 s pulses, while keeping fixture widths, material and support tensors fixed. Both 128-tetrahedron cases completed in **3.50 s total**, with no solver failures. Peak strains were **5.417% / 5.326%**, minimum J **0.99899 / 0.99921**, and peak nonrigid coordinate RMS **0.628 / 0.619 mm**. Rest moments were approximately **6.7×10⁻¹⁷ Nm** and peak actual dynamic moments **0.368 / 0.297 mNm**. Alignment works; zero dynamic torque was never assumed.

Both cases **fail the unchanged ≥1 mm nonrigid excitation requirement**. The frozen selection rule therefore chooses neither: no fine-mesh runs, force search or time qualification follow. This is a positive assembly/alignment result and a negative experiment-design result. There is still no qualified localized Gaussian nonlinear forward model. Simple proportional scaling of the 0.4 s result suggests that reaching 1 mm would require roughly 1.59× amplitude and could take local strain beyond 8%; this is a rough local estimate, not a simulated new result. Distributed traction or a more suitable geometry should be designed explicitly rather than raising force blindly.

## Nonlinear inverse numerical check

A separate two-parameter nonlinear bar inverse checks numerical posterior agreement with known support location and law; its held 2 N response is force interpolation, not future-motion forecasting. The first numerical attempt completed its 36 independent finite-element/oracle checks and both adaptive CDF integrations, but stopped before sampling: the held-right standard deviation differed by 0.187 µm between tolerance levels, exceeding the fixed 0.1 µm gate. Its CDF difference was only 7.62×10⁻⁸ and mean difference at most 2.51 nm. This failed attempt remains preserved. A separately frozen continuation uses centered, sensor-scaled moment integrals and a common normalization, with the identical observation bytes and unchanged scientific thresholds. It passes in **38.85 s on two CPU threads**. Both integration tolerances agree; all six SMC runs (8,192/32,768 particles × three seeds), all 15 pair comparisons and eight prior-started MCMC chains pass. Maximum R-hat is **1.00082**, minimum monitored bulk/tail ESS exceeds **10,453**, and the largest SMC/reference CDF difference is **0.00779**. Independent finite-element checks cover 36 prior/truth and 56 posterior/held-load cells; maximum full-field error is below **6.3×10⁻¹³ m**.

The 90% reference intervals are **E = [5,600.54, 5,962.28] Pa** and **K = [220.714, 231.336] N/m**. The fixed truths **6,000 Pa and 220 N/m both fall outside**. At the held 2 N force, posterior mean displacement errors are **0.239 mm left / 0.107 mm right**; the left truth falls outside its 90% latent interval and the right truth falls inside. These are one fixed noisy realization, not empirical coverage estimates. Numerical agreement does not establish population calibration or physical model fidelity. [Full nonlinear reference evidence](GAUSSIAN_NONLINEAR_BAR_INVERSE.md).

No new neural training or active-probing comparison is justified by these diagnostics alone. The earlier 32-object prediction results and their uncertainty failures remain unchanged.
