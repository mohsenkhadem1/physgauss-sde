# Local Bird sampler audit and correction

The 20 October 2026 four-chain, 8,000-draw reflected/interwoven run fails
88 of 733 monitors. Its maximum Rhat is 1.0339864; minimum bulk/tail ESS is
134.50/33.23. The old samples are discarded proposal-training data for this
experiment. No withheld particle positions inform the audit or proposal.

## Variables that still fail

The complete list, human-readable physical indices, original diagnostics,
input/reference/checkpoint hashes and exact cached-density replay are in
[`audit.json`](../examples/bird-local-sampler-audit/audit.json).

| Group | Failed monitors | Interpretation |
|---|---:|---|
| Physical parameters and path | 49 | Early rigid modal motion, late deformation, damping and mechanics discrepancy |
| Gaussian-bridge coordinates | 30 | The remaining state correlations, plus parameter counterparts |
| Conditional Gaussian load means | 8 | Load/path dependence remains poorly explored |
| Covariance normalizers | 1 | `logdet_Q`, associated with mechanics discrepancy |

The largest Rhat belongs to `physical_67`: frame 3 displacement, mode 5
(1.0339864, bulk ESS 172.32). The smallest tail ESS belongs to `native_236`:
the frame-10/mode-6 bridge displacement coordinate (33.23), closely followed
by physical frame-10 displacement mode 6 (34.69). Initial modes 0–5 are the
infinitesimal rigid modes; 6–11 are deformation modes. These are coordinates
of a fixed linear modal approximation, not independent real-world motions.

Log damping fails its tail ESS (311.80), and mechanics discrepancy fails
Rhat/bulk ESS (1.01588/308.44). Stiffness, smooth-load scale, RGB-error scale
and the overall log target pass their existing screens. A passing log-target
trace therefore does not establish path mixing.

![Discarded chain traces](../examples/bird-local-sampler-audit/failed-chain-traces.png)

## Fixed proposal correction

The previous floor-free guide learned only a five-parameter covariance and
left state covariance equal to identity. The actual 32,000 native bridge
draws have covariance eigenvalues 0.005721–17.814 (condition ratio 3,113.69).
The new fixed reference uses their full mean and covariance, including
state/parameter cross-correlation, with 2% shrinkage towards its empirical
diagonal and an absolute 0.001 ridge. All training draws are discarded.

The existing conditional Schur factorization preserves physical parameters
during state updates. Its dense state transform also transforms the linear
reflection walls. Exact reflected Hamiltonian state moves and both centered
and noncentered parameter slice moves retain the original floor-conditioned
hierarchical physical density. Gaussian load integration, all state-dependent
covariance determinants, guide Jacobians, observed pixels and floor likelihood
remain unchanged. The outer affine Jacobian is constant and cancels from
acceptance ratios. The new reference is a proposal, not a posterior mode.
Initial physical paths are preserved within 2.14e-14.

## Local CPU and GPU execution

This machine has an 18-core Intel i9-9980XE and four RTX 2080 Ti GPUs.
Mechanics, bridge factorization, reflection geometry and random streams remain
on CPU. The optional device adapter moves the full float64 floor-aware renderer
and its differentiable pose inputs to a selected GPU, then returns RGB to CPU.
It changes neither pixel selection nor precision. On actual Bird endpoints,
RGB differences are below 7e-16 and maximum pose-gradient differences below
6e-14. The full target on 16 stratified historical states agrees within
3.64e-12; see `cpu-cuda-target-replay.json` beside the audit.

The isolated environment uses Python 3.12.3, Torch 2.14.1+cu130 and
opt-einsum 3.4.0. The full regression invocation passed 192 tests in 154.77s;
the three added composition/device/index tests also passed separately.

```sh
python experiments/bird_local.py prepare \
  --gate runs/bird-8000/convergence.json \
  --reference runs/reference-original/reference.pt \
  --output runs/bird-preconditioned-reference

python experiments/bird_local.py run \
  --reference runs/bird-preconditioned-reference/reference.pt \
  --output runs/bird-local-pilot/chain-0 \
  --chain-index 0 --seed 1070009 --device cuda:0 \
  --warmup 128 --draws 256
```

Repeat the pilot for indices 1–3, seeds 1070010–1070012, and devices 1–3.
An initial CPU-only chain-3 trial was stopped after its 25-step warmup
checkpoint because it was substantially slower; its checkpoint is preserved
separately and its draws are not pooled with the GPU trial.

Pilot results are development diagnostics only. Fresh full chains require at
least 500 discarded warmup steps and 2,000 retained draws each, four distinct
seeds, exact checkpoint replay, Rhat <1.01 and bulk/tail ESS >=400 for every
physical/native/load/normalizer monitor, and zero retained invalid/divergent
state transitions. Independent prediction-score replay and the original
development readiness checks follow convergence. No wider comparison is
eligible before both gates pass. Passing one Bird case cannot establish
independent uncertainty calibration.

Current status: the local preconditioned pilot is running. No new convergence
or uncertainty improvement is claimed at this stage.
