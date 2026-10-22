# Fixed96tet Q8 backend bridge: source and preparation

This preparation snapshot contains no trajectory result or live-output hash. The reviewed protocol compares a newWarp96tet/Q8 trajectory with the alreadycompleted tightTorch96tet/Q8 reference. It isolates the backend before any surface-Q change. Material, fullconsistentM, contactlaw,clock/initializer and1e-10 numericalsettings remain fixed. Strict implementation gates remain10nm/10µm/s, alongside all original energy/P/J/mass checks. It does not qualify surfaceQ, meshresolution or rigidcontact.

v01 andv02 protocols were never executed and are retained with explicit supersession receipts. v01 lacked promisedresource/stopguards; v02 bounded ownfiles but omitted the requestedfilesystemFREEreserve. v03 is the corrected reviewed source: ownGPUresident/Torch8GiB, RSS12GiB, globalfree−projectedwrite>=30GiB (plusownfilecap30GiB), inheritedCPU2affinity/threads, owned1500s worker deadline andcontrollerTERM/INT cleanup. Checkpoints preserve the fullcommitted y/f/next-h prefix. Up to511 uncommitted acceptedtailsteps may be lost after hardkill.

Eight pureCPU/config/resource/ownedcleanup tests passed5.16s. The inventory deliberately excludes live execution/progress/resource logs/results, rawNPZ and compiledcache. Publication of the source protocol is not a claim that its live physical comparison has passed.
