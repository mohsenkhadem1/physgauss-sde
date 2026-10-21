# Neural boundary and material study

The [formulation](../../docs/GAUSSIAN_NEURAL_ACTIVE_FORMULATION.md) describes
physical Gaussian support/load fields, a direct neural trajectory operator,
conditional physical posteriors and acquisition by predictive-risk reduction.
The [baseline audit](../../docs/GAUSSIAN_BOUNDARY_BASELINES.md) distinguishes
the independent PIDG-inspired comparator from an official reproduction.

Final evaluation is in progress. Development gains do not establish recovered
physical parameters or calibrated uncertainty. The complete raw development
snapshot, including negative experiments, is published in
[the artifact index](artifacts.json). That index contains immutable commit URLs,
checksums and file counts. Completed simulation data and final test evidence
will be added separately.

To restore a published archive, download its `manifest_url` and use its
`download_base_url` from the index:

```sh
python experiments/restore_research_artifacts.py \
  --manifest manifest.json --parts downloaded-parts --destination restored \
  --download-base-url URL_FROM_ARTIFACT_INDEX
```

The restorer verifies every part, the combined archive and every restored file.
It refuses to overwrite a different existing file. Large raw artifacts are on
the artifact branch; source, protocols and compact reports belong on `main`.
