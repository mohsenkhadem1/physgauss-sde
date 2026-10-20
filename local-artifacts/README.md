# Local Bird raw checkpoints

This branch permanently preserves the observed-only development evidence and its original training artifact. None of these discarded draws enter the fresh full posterior.

## Original 8,000-draw diagnostic artifact

The original ZIP from Actions run 39415918136, artifact 12268611805, is split losslessly into 41 pieces. Its SHA-256 and size match GitHub's recorded artifact digest: 81,465,907 bytes; `c9b1e4999a0488421bcb9815b93bb0f6e2211801c65ff1b86fc7e544df304819`.

```sh
cat parent-8000.zip.part-* > parent-8000.zip
sha256sum parent-8000.zip
unzip parent-8000.zip -d parent-8000
```

The ZIP contains the complete 32,000-draw merged tensor, convergence report and every scalar diagnostic. `parent-8000-manifest.json` identifies every part. This diagnostics artifact does not contain the historical per-worker RNG checkpoints; it is sufficient to reproduce the proposal training together with the original reference and prepared inputs below.

## New discarded development trials

`development.zip` contains all new pilot chains, warmup checkpoints, fitted reference, recovery trials (including the rejected nondeterministic prototype), original reference, exact prepared inputs and logs. It contains no production posterior draws.

```sh
cat development.zip.part-* > development.zip
sha256sum development.zip
unzip development.zip
```

Expected SHA-256: `7da4c56da455e4913ce7520ee63b8667701ef2641597476be677e870c82ed6e2`.

The development manifest identifies all 101 original files and each part. The archive contains its own file manifest. Uploaded Git blob IDs for both archives were checked against local `git hash-object`; ZIP hashes were preserved. The code and compact reports are on main, and the fresh numerical worker is pinned at 4f0f0fcbfd518bd7be9b77001f5b23f34d5541dc.
