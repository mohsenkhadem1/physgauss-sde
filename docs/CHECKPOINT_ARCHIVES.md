# Public PAC-NeRF checkpoint archives

Permanent public publication of prepared inputs and full checkpoint ZIPs was
explicitly approved by the user on 18 October 2026. Compute remains on free
GitHub CPU workers.

- [Corrected exact-integral study](https://github.com/mohsenkhadem1/physgauss-sde/releases/tag/pacnerf-exact-2026-10-18): worker `219665b`, run `39100239747`. Inputs are published first; available completed-case results and checkpoint prefixes are added incrementally. A completion hook restricted to this run archives the remaining available artifacts.
- [SUPERSEDED quadrature experiment](https://github.com/mohsenkhadem1/physgauss-sde/releases/tag/pacnerf-historical-quadrature-2026-10-18): original worker `b2d081a`, run `39073611351`. A confirmed numerical integration defect makes these historical assets unsuitable as corrected-model results. They are preserved for reproduction and audit only.

These are experimental prereleases, not stable software releases or claims of
posterior convergence. File names contain original Actions artifact IDs.
Available checkpoints may be unfinished, unconverged, or references/warmup
states; consult each case's settings, diagnostics and the current result report.
Never combine a historical reference/chain with the corrected target.

Every uploaded ZIP must match the downloaded Actions archive's SHA256 and byte
count. Existing release assets must match GitHub's SHA256 and are never replaced
or deleted. Small manifests under `archives/` record permanent download links,
source commits, run IDs and digests. Original source attribution:
[PAC-NeRF](https://github.com/xuan-li/PAC-NeRF).

To reproduce an archive operation, download the matching run's `pacnerf-inputs`,
`pacnerf-results-*` and `pacnerf-chains-*` artifacts with the repository's
collector. Then run `scripts/archive_pacnerf_checkpoints.py --study exact`
(or `historical`) with the matching `--raw` and `--inputs` directories. The
default is a local plan with no authentication/network. The explicit
`--publish-public-archive` flag performs the already-approved publication.
