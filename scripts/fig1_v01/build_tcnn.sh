#!/bin/bash
# Build tiny-cuda-nn torch bindings into the M-PhyGs env with the local CUDA 12.1 toolchain (sm_75).
set -eu
TOOL=/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env
ENV=/home/bayespc2/vscode/guassianSDE/runs/gap-crossmethod-v02/mphygs-env
export CUDA_HOME=$TOOL CC=$TOOL/bin/x86_64-conda-linux-gnu-gcc CXX=$TOOL/bin/x86_64-conda-linux-gnu-g++
export TCNN_CUDA_ARCHITECTURES=75 TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=6
export CPLUS_INCLUDE_PATH=$TOOL/targets/x86_64-linux/include:$(ls -d $TOOL/lib/python3.10/site-packages/nvidia/*/include 2>/dev/null | tr '\n' ':')
export LIBRARY_PATH=/home/bayespc2/vscode/guassianSDE/runs/fig1-v01/tcnn-linklib:${LIBRARY_PATH:-} LD_LIBRARY_PATH=/home/bayespc2/vscode/guassianSDE/runs/fig1-v01/tcnn-linklib:$TOOL/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
export PATH=$ENV/bin:$TOOL/bin:$PATH
$ENV/bin/python -m pip install "setuptools<70" wheel ninja
$ENV/bin/python -m pip install --no-build-isolation -v "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
$ENV/bin/python -c "import tinycudann as t, torch; print('tcnn ok', t.__file__)"
