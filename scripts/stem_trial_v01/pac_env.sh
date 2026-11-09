#!/bin/bash
# PAC-NeRF runtime environment (same as scripts/gap_crossmethod_v02/pacnerf_launch.py), GPU chosen by caller.
TOOL=/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env; B=/home/bayespc2/vscode/guassianSDE/runs/gap-crossmethod-v02
export CUDA_HOME=$TOOL CC=$TOOL/bin/x86_64-conda-linux-gnu-gcc CXX=$TOOL/bin/x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR=$B/pacnerf-torch-extensions
export CPLUS_INCLUDE_PATH=$TOOL/targets/x86_64-linux/include:$(ls -d $TOOL/lib/python3.10/site-packages/nvidia/*/include | tr '\n' ':')
export PATH=$B/pacnerf-env/bin:$TOOL/bin:$PATH PYTHONPATH=$B/vendor/pac-nerf:/home/bayespc2/vscode/guassianSDE PYTHONUNBUFFERED=1
exec $B/pacnerf-env/bin/python -u "$@"
