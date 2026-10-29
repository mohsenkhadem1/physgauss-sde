"""Development check of affine-preserving transfer with smooth finite attachments."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.study import BASE,OUT,within_deadline
from scripts.phystwin.sequential_reference import write
settings=json.loads((BASE/'method-candidate.json').read_text());directory=BASE/settings['mpm']['tag']/'development-1';fit=json.loads((directory/'summary.json').read_text());baseline=np.load(directory/'predictions.npz');rows=[];fields={}
for rpic in [0.,.5,-1.]:
 for dt in [.0001,.00005]:
  if rpic==-1. and dt==.00005:continue
  m=PositionMPM(2,64,dt,rpic=rpic,attachment_width=.006,ramp_seconds=1.);m.parameters(**parameters(fit['best']['z'],fit['best']['layout']))
  for seconds in [4.,8.,16.]:
   if rpic==-1. and seconds==4.:continue
   for action in [0,6]:
    within_deadline()
    try:
     s=m.solve(baseline['poses'][action],seconds=seconds);u=s.pop('field');key=f'r{rpic}-dt{dt}-s{seconds}-a{action}';fields[key]=u;s.update(rpic=rpic,dt=dt,seconds=seconds,action=action,valid=True)
     previous=f'r{rpic}-dt{dt}-s{seconds/2}-a{action}'
     if previous in fields:s['doubling_change_mm']=float(1000*np.sqrt(np.mean((u-fields[previous])**2)))
    except RuntimeError as e:s=dict(rpic=rpic,dt=dt,seconds=seconds,action=action,valid=False,error=str(e))
    rows.append(s);write(OUT/'mpm-affine-transfer-audit.json',dict(rows=rows,scope='Fixed development parameters; affine transfer and longer holds test whether numerical filtering biases positional response. No held-out method scores used.'));np.savez_compressed(BASE/'mpm-affine-transfer-audit.npz',**fields);print('AFFINE_MPM',s,flush=True)
  del m
