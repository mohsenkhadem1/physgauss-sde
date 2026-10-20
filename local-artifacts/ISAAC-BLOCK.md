# Isaac Sim boundary block: raw development evidence

Reassemble `cat isaac-block-development.tar.xz.part-* > isaac-block-development.tar.xz`, verify SHA256 ee1bcddceb28fbf9d1e2700d249f62a2776b39e4fb981c8a91679c89fee7ef18, then `tar -xJf isaac-block-development.tar.xz`.

This archive preserves all461 files: exact scenes, references, Gaussian/nodal trajectories, force schedules,61 actual RGB frames per recorded episode, source snapshots, runtime freeze and logs, including interrupted/superseded trials. Every decompressed file hash and uploaded Git blob SHA was checked.

Only `runs/isaac-boundary-block-v3*` cases have the final explicit-clock/source/force/geometry checks. Three geometry/force tests and five final GPU episodes pass their integrity checks. No boundary inference or calibrated uncertainty is implemented. The240/480Hz paths differ15.08mm coordinate RMS and134.13mm maximum node separation; numerical convergence is NOT established. Treat this as a fixed-discretization development simulator, not a calibrated force benchmark. Earlier trials are marked development only. Bird remains stopped by explicit user instruction.
