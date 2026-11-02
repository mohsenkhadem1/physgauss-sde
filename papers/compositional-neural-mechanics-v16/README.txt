Working paper scope

Candidate title: Compositional Neural Mechanics with Observable Physical Uncertainty

Central question: when can a visual reconstruction predict the correct physical response, and which additional observations constrain that response?

Candidate contribution: reusable local bulk/interface energies composed by a mechanical solver, with explicit scene descriptors and inference that respects physical ambiguities. Individual ingredients (neural energies, Gaussian support fields, differentiable simulation and active sensing) are not claimed to be independently novel.

Current evidence belongs to development, not the final test:
- Mapped-block nonlinear material/support/geometry combinations.
- Verified common-stiffness scaling and numerical mesh checks.
- Attachment extrapolation failure and a separately audited repair.
- Controlled position-only ambiguity, resolution by reaction measurements, and resolution by known gravity/density.
- A larger continuous-parameter inference study and a sealed oracle forward benchmark are in progress.

Paper organization:
1. Visual agreement versus functional physical accuracy; define future deformation/reaction targets.
2. Material coordinates, objective shared constitutive law, relational interface energy and solver composition.
3. Scene inference and exact common-scale ambiguity; separate known mass/load assumptions.
4. Oracle forward transfer, inferred-descriptor transfer, uncertainty coverage, and fixed/random/adaptive acquisition at matched budgets.
5. Scope and measured failures, including finite contact penetration and unresolved small contact patches.

Report all comparison variants. The matched analytic teacher is an oracle ceiling; Neo-Hookean is deliberately misspecified but small-strain matched. The direct predictors are custom baselines, not implementations of authors' MeshGraphNet or PhysTwin systems. Include their training/validation gap. Existing real-dataset benchmark results are preserved but cannot be attributed to this new representation without adapting and rerunning it.

No completed-paper or state-of-the-art claim is supported by this draft. Populate the final tables only from the frozen reports. The detailed formulation is reports/compositional-neural-mechanics-v16/FORMULATION.txt.
