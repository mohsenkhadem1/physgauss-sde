"""Read-only independent Kabsch decomposition of native MASIV diagnostic pairs."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
B=Path('runs/gap-crossmethod-v02/masiv');rows=[]
def aligned(a,b):
 x=a-a.mean(0);y=b-b.mean(0);u,_,vt=np.linalg.svd(x.T@y);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);return np.sqrt(np.mean(np.sum((x@(u@d@vt)-y)**2,axis=1)))
for folder in [B,B/'contact-v02']:
 for p in sorted(folder.glob('*scale1*.npz')):
  other=p.with_name(p.name.replace('scale1','scale2'))
  if not other.exists():continue
  a=np.load(p)['x'].astype(float);b=np.load(other)['x'].astype(float)
  e=np.linalg.norm(a-b,axis=-1);res=np.array([aligned(x,y) for x,y in zip(a,b)])
  rows.append(dict(case=str(p.relative_to(B)),raw_trajectory_point_RMS_mm=float(np.sqrt(np.mean(e*e))*1000),raw_final_point_RMS_mm=float(np.sqrt(np.mean(e[-1]**2))*1000),rigid_aligned_final_point_RMS_mm=float(res[-1]*1000),rigid_aligned_trajectory_point_RMS_mm=float(np.sqrt(np.mean(res*res))*1000),source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [p,other]}))
files=['scripts/gap_crossmethod_v02/masiv_native_forward.py','scripts/gap_crossmethod_v02/masiv_render_diagnostic.py','runs/gap-crossmethod-v02/vendor/masiv/nclaw/sim/mpm.py','runs/gap-crossmethod-v02/vendor/masiv/nclaw/material/meta.py']
out={'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},'scope':'Geometry comparison after optimal rotation and translation; no scaling removed. Independent decomposition, not a fitted native method benchmark.'}
Path('reports/gap-crossmethod-v02/masiv-independent-geometry-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(rows,indent=2))
