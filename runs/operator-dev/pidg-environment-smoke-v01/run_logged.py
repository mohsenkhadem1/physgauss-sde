import subprocess,sys,json,datetime,os
from pathlib import Path
root=Path(__file__).resolve().parent
name=sys.argv[1];cmd=sys.argv[2:]
env=os.environ.copy();env.update(MAMBA_ROOT_PREFIX=str(root/'mamba-root'),PIP_CACHE_DIR=str(root/'pip-cache'),CUDA_VISIBLE_DEVICES=os.environ.get('PIDG_SMOKE_GPU',''),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
env.update(PATH=str(root/'env/bin')+':'+env.get('PATH',''),CUDA_HOME=str(root/'env'),CC=str(root/'env/bin/x86_64-conda-linux-gnu-gcc'),CXX=str(root/'env/bin/x86_64-conda-linux-gnu-g++'),CUDAHOSTCXX=str(root/'env/bin/x86_64-conda-linux-gnu-g++'))
cuda_headers=':'.join([str(root/'env/targets/x86_64-linux/include')]+[str(p) for p in sorted((root/'env/lib/python3.10/site-packages/nvidia').glob('*/include'))])
env.update(CPATH=cuda_headers,CPLUS_INCLUDE_PATH=cuda_headers,LIBRARY_PATH=str(root/'env/targets/x86_64-linux/lib'))
assert env['CUDA_VISIBLE_DEVICES'] in ('','1'), 'Only reserved GPU1 permitted'
env.update(MPLCONFIGDIR=str(root/'mpl-config'),MPLBACKEND='Agg',XDG_CACHE_HOME=str(root/'cache'),TORCH_HOME=str(root/'torch-cache'))
env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_DEVICE_ORDER='PCI_BUS_ID')
record={'command':cmd,'cwd':os.getcwd(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'scoped_environment':{k:env[k] for k in ['CUDA_HOME','CC','CXX','CUDAHOSTCXX','CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','MAX_JOBS','TORCH_CUDA_ARCH_LIST','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_DEVICE_ORDER']}}
p=root/'logs'/(name+'.json');p.write_text(json.dumps(record,indent=2)+'\n')
with (root/'logs'/(name+'.log')).open('w') as f:
 proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);record['pid']=proc.pid;p.write_text(json.dumps(record,indent=2)+'\n');proc.wait()
record.update(exit_code=proc.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));sys.exit(proc.returncode)
