"""Development-only time/grid/particle/settling sensitivity of the MPM adapter."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_reference import write
obs=np.load(BASE/'observations/development-1.npz');r=json.loads((BASE/'mpm-layer0.006/development-1/summary.json').read_text());rows=[];fields={}
# Reuse the same fitted physical parameters; this tests numerical sensitivity,
# independently of any held-out prediction score or subsequent refitting.
for name,res,grid,dt,seconds in [('base',2,64,.0001,4.),('time',2,64,.00005,4.),('grid',3,80,.00005,4.),('settle',2,64,.0001,8.)]:
 m=PositionMPM(res,grid,dt,rpic=-1.,attachment_width=.006,ramp_seconds=1.);m.parameters(**parameters(r['best']['z'],r['best']['layout']))
 for action in [0,4,7,8]:
  within_deadline()
  try:
   s=m.solve(obs['poses'][action],seconds=seconds,block=20);u=s.pop('field');fields[f'{name}-{action}']=u;s.update(action=action,configuration=name,resolution=res,grid=grid,dt=dt,seconds=seconds,numerically_valid=True,settled=bool(s['drift_m']<2e-5))
   if name!='base':s['change_from_base_rms_m']=float(np.sqrt(np.mean((u-fields[f'base-{action}'])**2)))
  except RuntimeError as e:s=dict(action=action,configuration=name,numerically_valid=False,settled=False,error=repr(e))
  rows.append(s);write(BASE/'mpm-resolution-audit.json',rows);np.savez_compressed(BASE/'mpm-resolution-audit.npz',**fields);print('MPM_RESOLUTION',s,flush=True)
 del m
