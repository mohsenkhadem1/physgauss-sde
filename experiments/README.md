# Bird sampler development

The current scientific direction is [boundary-condition learning](../docs/BOUNDARY_LEARNING_DIRECTION.md).
Bird sampler experiments provide inference diagnostics; they do not yet learn
unknown fixture locations, boundary types or contact hypotheses.

`bird_collapsed_state.py` is an unintegrated constant-noise prototype for exact
Gaussian velocity/load elimination and conditional refreshes. Three tests
check density factorization, derivatives and position/floor geometry. No full
transition recovery or convergence result is available; do not use it to resume
or replace the active frozen worker.

The production run is recorded in `configs/pacnerf_bird_check.json`. Its
convergence and uncertainty gates take precedence over development diagnostics.

`bird_joint_hmc.py` tests full joint HMC on the same floor-conditioned posterior,
using a previously saved observed-data sampling reference. It moves mechanics,
noise parameters and the hidden state together. The production hybrid kernel
instead alternates state HMC and scalar parameter elliptical slices.

Run from the repository root after installing the project:

```sh
python experiments/bird_joint_hmc.py --inputs runs/pacnerf-inputs --reference path/to/reference.pt --output runs/new-joint-development --draws 256
```

This diagnostic fixes four chains,100 warmup steps, seed590009, randomized
Hamiltonian time1–4 and target acceptance0.9. Input/reference hashes and source
are frozen before sampling. Increase `--draws` with the same output directory
to continue the exact saved RNG/kernel. Start with an empty directory. No
future particle positions or held-out images are scored. Its output is never
eligible for production ranking: the production gate has longer warmup and
also checks load/density diagnostics and independently replays predictions.

The initial local exploratory wrapper used these same kernel settings and the
learned reference at `runs/floor-adapted-hybrid-development/reference.pt`.
That local run is exploratory; it does not replace the formal GitHub run.
