# Discarded conditional-transport development

The 128/256-draw four-chain pilot is not converged and is not promoted. At 256 draws worst Rhat is 1.09703, minimum bulk/tail ESS 33.16/85.34; zero retained invalid/divergent steps. Density replay and saved-prefix preservation pass. The physical posterior is unchanged. No withheld geometry was evaluated.

The archive contains both checkpoints (including the original 128-draw RNG state), learned proposal weights/reference, full diagnostics, logs, paired long-trajectory probes and exact development source. Every archived file and uploaded Git blob was independently hash checked.

```sh
cat conditional-development.tar.xz.part-* > conditional-development.tar.xz
sha256sum conditional-development.tar.xz
tar -xJf conditional-development.tar.xz
```

Expected SHA-256: `c1f3ea6760899bdf63d896a9ea4c1a87600ade292f318f61ced29cbaae290146` (3,131,568 bytes). These discarded states are never pooled with the independently seeded full run.
