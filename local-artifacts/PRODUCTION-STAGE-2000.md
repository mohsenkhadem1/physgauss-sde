# Bird full run: 2,000-draw stage (not converged)

Four independently seeded chains completed 500 discarded warmup steps and 2,000 retained draws each. The strict 733-monitor gate FAILS: 45 quantities fail, worst Rhat 1.012508, minimum bulk/tail ESS 256.58/131.34. There are zero retained invalid/divergent state steps; density replay passes. The unchanged chains continue separately to the predeclared next budget. No predictive scores were opened.

This archive preserves all four exact per-worker RNG checkpoints, references, frozen protocols and the full raw merged tensor/diagnostic reports. Every archived file was decompressed and SHA-256 checked against its source. Uploaded Git blob IDs also match local git hash-object.

```sh
cat production-stage-2000.tar.xz.part-* > production-stage-2000.tar.xz
sha256sum production-stage-2000.tar.xz
tar -xJf production-stage-2000.tar.xz
```

Expected SHA-256: `d4cc1e14215b1a70cbee453a3fd77d25f373174acf06f8310292561eb9eabe1d` (20,606,304 bytes). See the stage manifest and embedded ARCHIVE-MANIFEST.json for file hashes and worker source. These samples are retained in exact continuations; they are not pooled with any development pilot.
