from pathlib import Path
import os,subprocess
root=Path.cwd();base=Path('/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env');env=dict(os.environ,CUDA_HOME=str(base),CC=str(base/'bin/x86_64-conda-linux-gnu-gcc'),CXX=str(base/'bin/x86_64-conda-linux-gnu-g++'),FORCE_CUDA='1',TORCH_CUDA_ARCH_LIST='7.5',MAX_JOBS='2',CPLUS_INCLUDE_PATH=str(base/'targets/x86_64-linux/include')+':'+':'.join(str(p) for p in (base/'lib/python3.10/site-packages/nvidia').glob('*/include')))
for project in ['pytorch3d','mip-splatting/submodules/diff-gaussian-rasterization','utils3d','nvdiffrast']:
 log=(root/'runs/gap-first-v01/drop'/('build-'+project.split('/')[0]+'.log')).open('w');p=subprocess.run(['uv','pip','install','--python','runs/gap-first-v01/drop/env/bin/python','--no-build-isolation','--no-deps','./runs/gap-first-v01/drop/vendor/'+project],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();print(project,p.returncode,flush=True)
