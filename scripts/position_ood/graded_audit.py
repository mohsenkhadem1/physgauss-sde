"""Independent graded-mesh development convergence diagnostic."""
from pathlib import Path
import json,sys,time,gc
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.graded import GradedMaterialFixture
from scripts.position_ood.study import BASE,OUT,protocol,within_deadline
from scripts.phystwin.sequential_reference import write
rows=[];case=protocol()['development'][1];ref=np.load(BASE/'fields/development-1.npz');previous=None
for n in [2,4,6,8]:
 within_deadline();m=GradedMaterialFixture(n,**case);fields=[];u=None;start=time.time()
 for action in [0,4,6]:
  m.set_controls(ref['poses'][action]);state=m.solve(initial=u,force_tolerance_N=1e-9);u=state['displacement_local_m'];fields.append(m.sampled(state)['field']);print('GRADED',n,action,flush=True)
 fields=np.array(fields);row=dict(mesh=n,actions=[0,4,6],error_against_uniform_n10_mm=(1000*np.sqrt(np.mean((fields-ref['field'][[0,4,6]])**2,axis=(1,2)))).tolist(),seconds=time.time()-start)
 if previous is not None:row['adjacent_change_mm']=(1000*np.sqrt(np.mean((fields-previous)**2,axis=(1,2)))).tolist()
 rows.append(row);write(OUT/'graded-development-audit.json',dict(scope='Development-only mesh-design diagnostic; uniform n10 is itself a finite reference, not exact truth.',rows=rows));np.savez_compressed(BASE/f'graded-development-1-n{n}.npz',fields=fields);previous=fields;m.clear_linear_cache();del m;gc.collect();print(row,flush=True)
