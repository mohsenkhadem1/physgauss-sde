"""Long-horizon endpoint drift with a fixed MPM physical fit."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_reference import write
obs=np.load(BASE/'observations/development-1.npz');r=json.loads((BASE/'mpm-layer0.006/development-1/summary.json').read_text());rows=[];fields={}
for dt in [.0001,.00005,.000025]:
 m=PositionMPM(2,64,dt,rpic=-1.,attachment_width=.006,ramp_seconds=1.);m.parameters(**parameters(r['best']['z'],r['best']['layout']))
 for seconds in [4.,8.,16.]:
  for action in [0,8]:
   within_deadline()
   try:
    s=m.solve(obs['poses'][action],seconds=seconds,block=50);u=s.pop('field');fields[f'dt{dt}-s{seconds}-a{action}']=u;s.update(action=action,dt=dt,seconds=seconds,numerically_valid=True)
    if seconds>4:s['doubling_change_rms_m']=float(np.sqrt(np.mean((u-fields[f'dt{dt}-s{seconds/2}-a{action}'])**2)))
   except RuntimeError as e:s=dict(action=action,dt=dt,seconds=seconds,numerically_valid=False,error=repr(e))
   rows.append(s);write(BASE/'mpm-settling-audit.json',rows);np.savez_compressed(BASE/'mpm-settling-audit.npz',**fields);print('MPM_SETTLING',s,flush=True)
 del m
