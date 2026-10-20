# Bird stopped by user request

The user requested stopping Bird and starting an Isaac Sim boundary-learning block. No Bird job remains active. The four last partial checkpoints retain 4150/4100/4150/4100 draws; the complete 4000 stage fails92/733 monitors (Rhat1.2264256; minimum bulk/tail ESS12.9310/10.5135). These are not converged posterior estimates. No predictive evaluation or wider comparison was opened.

Reassemble `cat production-user-stopped.tar.xz.part-* > production-user-stopped.tar.xz`, verify SHA256 5caae9576516bd4ca2393f3358296b59063876f9fd06a6d939932c567ef8de5f, then `tar -xJf production-user-stopped.tar.xz`. The archive preserves all83 files, including every saved partial worker/RNG checkpoint, immutable2000/4000 snapshots, raw previews, aggregate diagnostics and logs. Every decompressed file hash and uploaded Git blob ID was checked. Source worker remains4f0f0fcbfd518bd7be9b77001f5b23f34d5541dc. Do not restart Bird without a new user instruction.
