Dense mechanics integration on PIDG

The previous 13-node spherical equilibrium adapter severely reduced appearance quality. This experiment retains PIDG's dense canonical Gaussian appearance and deformation map, then adds the existing frozen ScaleSeparatedMaterialLaw and a 16-kernel Gaussian interaction field as a differentiable mechanics constraint. It is not an evaluation of the original coarse adapter, nor a claim that the full paper forward solver now matches native rendering.

On 384 deterministic opacity-ranked support points, a reference-neighborhood least-squares gradient estimates F. The learned material energy gives internal forces by differentiation. Central temporal differences give acceleration and velocity. Gaussian interactions combine temporally interpolated vector loads, compliant targets, positive stiffness/damping and a global body-load vector. A fixed reconstruction-derived acceleration scale defines nondimensional force units. Image gradients and mechanics gradients jointly update the retained dense deformation. Material weights remain frozen; scene descriptors and Gaussian loads are optimized.

The constraint is soft. The renderer still queries a learned temporal deformation map; it does not integrate a certified contact rollout. No force ground truth, SI scale, true elasticity identification or calibrated uncertainty is claimed. This phase tests whether adding our physical representation can preserve rendering quality across several PIDG scenes before attempting stronger forward coupling.

Protocol: 50k native base, then independently restarted 5k refinements from that same base for native and three positive mechanics weights. Same image set, seeds, appearance capacity and schedules in each arm. Highest validation foreground PSNR selects the hybrid; final test access follows a checkpoint-hash seal. Do not select by test results. All candidates remain in reports. Raw data and large checkpoints stay local.

Run prepare.py for each scene. Motion Kuro and Balls use training-only flow preprocessing; Rubber Duck follows its upstream no-flow configuration. run.py maps the release's rubber_duck name to the upstream rubber_dack configuration. The transform parser normalizes slash and .png variants before merging IDs.

Worker: python3 scripts/pidg_dense_mechanics/campaign.py --scene SCENE --gpu GPU
Summary after all three workers complete: python3 scripts/pidg_dense_mechanics/report.py
Figures: runs/phystwin-env/bin/python scripts/pidg_dense_mechanics/figures.py
Tests and identity checks use the isolated official PIDG Python environment.
