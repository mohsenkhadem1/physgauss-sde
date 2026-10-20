# Isaac boundary orbit dataset v0.1

48 retained episodes, 2,928 RGB/depth/mask frames, ten sparse observations per episode.
All raw evidence, preflights and the rejected GPU 3 capture are preserved.

Archive: isaac-boundary-orbit-v01.tar.xz (303470132 bytes), split into 152 ordered parts.
SHA256: 4a5c6580347069134a9387923d4ce92966fcfb49fc726bb406caaf75eab1f22c
Every one of 11,520 original files passed a byte-for-byte archive round trip.

Use experiments/restore_isaac_dataset.py from main with the index in
examples/isaac-boundary-orbit/raw-archive.json. It verifies every part and the
combined SHA256 before safely extracting. manifest.json contains all file hashes.

Protocol, results and videos:
https://github.com/mohsenkhadem1/physgauss-sde/tree/main/examples/isaac-boundary-orbit
