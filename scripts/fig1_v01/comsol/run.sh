#!/bin/bash
# run.sh ARGS... : compile-free launcher for Fig1Comsol (used inside a systemd unit)
D=/home/bayespc2/Documents/COMSOL64; export LD_LIBRARY_PATH=$D/ext/cadimport/glnxa64:$D/lib/glnxa64:$D/bin/glnxa64
cd /home/bayespc2/vscode/guassianSDE/scripts/fig1_v01/comsol
exec $D/java/glnxa64/jre/bin/java -Xmx${XMX:-12g} -Djava.library.path=$D/lib/glnxa64 -cp ".:$D/plugins/*" Fig1Comsol "$@"
