"""Launch pinned native GIC training durably on the assigned physical GPU1."""
from pathlib import Path
import os,json,time,subprocess
R=Path(__file__).resolve().parents[2]
report=R/'reports/gap-crossmethod-v02';status=report/'gic-worker.json'
if status.exists():
 old=json.loads(status.read_text())
 try:os.kill(old['pid'],0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Existing worker live')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[str(R/'runs/gap-crossmethod-v02/gic-env/bin/python'),'-u',str(R/'scripts/gap_crossmethod_v02/gic_native_entry.py'),'-c',str(R/'runs/gap-crossmethod-v02/gic-torus-prefix.json'),'-s',str(R/'runs/gap-crossmethod-v02/gic-inputs/torus'),'-m',str(R/'runs/gap-crossmethod-v02/gic-output/torus'),'--reg_scale','--reg_alpha']
log=open(report/'gic-train.log','ab',buffering=0)
p=subprocess.Popen(cmd,cwd=R/'runs/gap-crossmethod-v02/vendor/gic',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
status.write_text(json.dumps(dict(pid=p.pid,started_unix=time.time(),cmd=cmd,physical_gpu=1,scope='Native author fit, explicit held view/time split and CPU image storage'),indent=2));print(p.pid)
