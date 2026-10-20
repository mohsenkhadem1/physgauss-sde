# Local Bird raw checkpoints

The development archive contains all new discarded pilot chains, warmup checkpoints, fitted reference, raw convergence tensors, recovery trials (including the rejected nondeterministic prototype), original prepared inputs and logs. It contains no production posterior draws. The code and compact reports are on main.

Download every `development.zip.part-*` file without altering bytes, then run:

```sh
cat development.zip.part-* > development.zip
sha256sum development.zip
unzip development.zip
```

Expected SHA-256: `7da4c56da455e4913ce7520ee63b8667701ef2641597476be677e870c82ed6e2`.

`development-manifest.json` identifies the 101 original files and each part by size and SHA-256. `ARCHIVE-MANIFEST.json` inside the ZIP records original provenance. All 11 uploaded Git blob IDs were independently checked against local `git hash-object`. The historical 8,000-draw checkpoint used for proposal learning remains artifact 12268611805 of run 39415918136; the compact audit records its exact merged tensor hash. No discarded samples are pooled with the fresh full test.
