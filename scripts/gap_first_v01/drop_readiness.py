"""Bounded environment readiness gate; only four development frames, not final tests."""
import datetime,json,os,subprocess,time
from pathlib import Path
root=Path.cwd();base=root/'runs/gap-first-v01/drop';report=root/'reports/gap-first-v01/drop/readiness.json';deadline=datetime.datetime.fromisoformat('2026-11-07T11:44:00+00:00').timestamp();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(base/'vendor/ReconPhys'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
while time.time()<deadline:
 p=subprocess.run([str(base/'env/bin/python'),'-c','from pytorch3d.ops import knn_points; from trellis.utils import render_utils; print("native imports ready")'],env=env,capture_output=True,text=True)
 d=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),imports_ready=p.returncode==0,import_error=p.stderr[-2000:] if p.returncode else '',phase='dependency_gate')
 report.write_text(json.dumps(d,indent=2))
 if p.returncode==0:
  with (base/'pilot.log').open('w') as log:q=subprocess.run([str(base/'env/bin/python'),'scripts/gap_first_v01/drop_run.py','--asset','00e2a405dcff4701a8f57d2e67e306b8','--pilot'],env=env,stdout=log,stderr=subprocess.STDOUT)
  d.update(phase='pilot_finished',pilot_returncode=q.returncode);report.write_text(json.dumps(d,indent=2));break
 time.sleep(30)
else:report.write_text(json.dumps(dict(phase='two_hour_gate_blocked',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
