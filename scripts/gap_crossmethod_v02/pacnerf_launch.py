"""Isolated native PAC-NeRF runtime; no modifications to base environments."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-crossmethod-v02';tool=Path('/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_HOME=str(tool),CC=str(tool/'bin/x86_64-conda-linux-gnu-gcc'),CXX=str(tool/'bin/x86_64-conda-linux-gnu-g++'),TORCH_CUDA_ARCH_LIST='7.5',MAX_JOBS='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',TORCH_EXTENSIONS_DIR=str(BASE/'pacnerf-torch-extensions'),CPLUS_INCLUDE_PATH=str(tool/'targets/x86_64-linux/include')+':'+':'.join(str(p) for p in (tool/'lib/python3.10/site-packages/nvidia').glob('*/include')))
env['PATH']=str(BASE/'pacnerf-env/bin')+':'+str(tool/'bin')+':'+env['PATH']
env['PYTHONPATH']=str(BASE/'vendor/pac-nerf')+':'+str(ROOT)
os.execve(str(BASE/'pacnerf-env/bin/python'),[str(BASE/'pacnerf-env/bin/python'),'-u',*sys.argv[1:]],env)
