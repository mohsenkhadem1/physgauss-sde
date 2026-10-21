# Completed Isaac boundary dataset and sensor QA

272 episodes: original48 plus fresh128TRAIN/32VAL/64TEST. All61savedframes of every episode passed independent sensor-depth checks. Includes actual four-candidate camera pools and independent evaluation views for16prespecified TEST states. Known cuboid, supplied cameras/masks, fixed simulator discretization; no continuum-convergence claim.

Archive:1,912,729,951bytes,48parts. SHA256 f7f47f744185431758067b43288017b8564ff33148a8204b27de153a3a76bb46. All65,192files were stream-verified and restored to disk with every file hash checked. Original failed/partial attempts and exact-state sensor-repair provenance are retained.

Download manifest.json and use the checksum-verifying restore_research_artifacts.py from main. The main artifact index supplies immutable commit URLs. This archive alone contains simulation/infrastructure evidence, not final model-performance claims. Restore distinct archives into separate directories unless all overlapping files are identical; the restorer refuses different-file clobbering.
