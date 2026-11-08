"""Resume native GIC after the completed 40k appearance stage: velocity + physical-parameter fit.
Only change versus gic_launch.py: cameras resident on GPU (upstream default), because the
upstream prepare_gt/Estimator paths assume GPU cameras. Appearance checkpoint is reused, not retrained.
"""
from pathlib import Path
import os,json,time,subprocess
R=Path(__file__).resolve().parents[2]
report=R/'reports/gap-crossmethod-v02';status=report/'gic-physics-worker.json'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[str(R/'runs/gap-crossmethod-v02/gic-env/bin/python'),'-u',str(R/'scripts/gap_crossmethod_v02/gic_native_entry.py'),'-c',str(R/'runs/gap-crossmethod-v02/gic-torus-prefix-physics.json'),'-s',str(R/'runs/gap-crossmethod-v02/gic-inputs/torus'),'-m',str(R/'runs/gap-crossmethod-v02/gic-output/torus'),'--reg_scale','--reg_alpha']
log=open(report/'gic-physics.log','ab',buffering=0)
p=subprocess.Popen(cmd,cwd=R/'runs/gap-crossmethod-v02/vendor/gic',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
status.write_text(json.dumps(dict(pid=p.pid,started_unix=time.time(),cmd=cmd,physical_gpu=1,scope='Native GIC velocity+physics stages on frozen 40k appearance checkpoint; GPU-resident cameras'),indent=2));print(p.pid)
