"""Re-run full current-geometry development qualification (never held-out tuning)."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.scenarios import run
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,time,json,hashlib
NAMES=['squeeze','slide','moving_sphere','curved','edge_drop']
def task(q):
 name,cells,dt,E,mu=q;start=time.time()
 try:
  o=run(name,young=E,friction=mu,cells=cells,dt=dt)
  a=o['rest']-o['rest'].mean(0);nr=[]
  for frame in o['frames']:
   b=frame-frame.mean(0);u,_,vt=np.linalg.svd(a.T@b);R=u@vt
   if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vt
   nr.append(float(np.sqrt(np.mean((a@R-b)**2))))
  return dict(name=name,cells=cells,dt=dt,young=E,friction=mu,seconds=time.time()-start,min_j=o['min_j'],max_penetration_m=o['max_penetration_m'],friction_saturated_fraction=o['friction_saturated_fraction'],max_kabsch_nonrigid_coordinate_rms_m=max(nr),final_kabsch_nonrigid_coordinate_rms_m=nr[-1],frames=o['frames'],centers=o['centers'],times=o['times'])
 except Exception as e:return dict(name=name,cells=cells,dt=dt,young=E,friction=mu,error=str(e),seconds=time.time()-start)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--workers',type=int,default=4)
 parser.add_argument('--output',type=Path,default=ROOT/'reports/coupled-contact-v13/physics-development-qualification-rerun.json')
 args=parser.parse_args()
 sources=[ROOT/'scripts/coupled_contact_v13'/f for f in ['physics.py','scenarios.py','qualification.py']]
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
 qs=[(n,cells,dt,6000.,.35) for n in NAMES for cells,dts in [((2,2,2),[2e-4,1e-4,5e-5]),((4,4,4),[1e-4,5e-5,2.5e-5])] for dt in dts]+[(n,(4,4,4),5e-5,3000.,.1) for n in NAMES]
 rows=[]
 with ProcessPoolExecutor(max_workers=max(1,min(args.workers,4))) as ex:
  for r in ex.map(task,qs):
   rows.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('frames','centers','times')}),flush=True)
 comparisons=[]
 for n in NAMES:
  for cells in ((2,2,2),(4,4,4)):
   group=[r for r in rows if r['name']==n and tuple(r['cells'])==cells and r['young']==6000.]
   out=dict(name=n,cells=cells)
   if any('error' in r for r in group):out.update(error='simulation failure',gates=dict(finite=False))
   else:
    a,b,c=group
    assert np.allclose(a['times'],c['times']) and np.allclose(b['times'],c['times'])
    errs=[float(np.sqrt(np.mean((r['frames']-c['frames'])**2))) for r in (a,b)]
    rigiderrs=[float(np.sqrt(np.mean((r['centers']-c['centers'])**2))) for r in (a,b)]
    ratio=errs[1]/max(errs[0],1e-20)
    out.update(timesteps=[r['dt'] for r in group],alltime_coordinate_rms_to_finest_m=errs,alltime_collider_center_rms_to_finest_m=rigiderrs,refinement_ratio=ratio,gates=dict(finite=True,penetration_below_1mm=max(r['max_penetration_m'] for r in group)<=.001,medium_to_finest_below_point25mm=errs[1]<=.00025,refinement_ratio_below_point8=ratio<=.8))
   comparisons.append(out)
 report=dict(scope='Development-only deterministic numerical qualification; not held-out results',source_hashes=hashes,runs=[{k:v for k,v in r.items() if k not in ('frames','centers','times')} for r in rows],comparisons=comparisons,all_gates_pass=all(all(c['gates'].values()) for c in comparisons))
 report['soft_development_gates']={r['name']:dict(finite=('error' not in r),positive_j=r.get('min_j',0)>.02,penetration_below_1mm=r.get('max_penetration_m',float('inf'))<=.001) for r in rows if r['young']==3000.}
 report['all_gates_pass']=report['all_gates_pass'] and all(all(r.values()) for r in report['soft_development_gates'].values())
 report['source_unchanged_during_run']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
 report['all_gates_pass']=report['all_gates_pass'] and report['source_unchanged_during_run']
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(comparisons,indent=2),flush=True)
