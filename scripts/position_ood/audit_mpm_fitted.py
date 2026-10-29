"""Timestep and hold-duration checks at a fitted development MPM explanation."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.study import BASE,OUT,within_deadline
from scripts.phystwin.sequential_reference import write
settings=json.loads((BASE/'method-candidate.json').read_text());directory=BASE/settings['mpm']['tag']/'development-1';fit=json.loads((directory/'summary.json').read_text());baseline=np.load(directory/'predictions.npz');rows=[];fields={}
for dt,seconds in [(.0001,8.),(.00005,4.)]:
 model=PositionMPM(2,64,dt,rpic=-1.,attachment_width=.006,ramp_seconds=1.);model.parameters(**parameters(fit['best']['z'],fit['best']['layout']))
 for action in [0,4,6]:
  within_deadline();r=model.solve(baseline['poses'][action],seconds=seconds);u=r.pop('field');fields[f'dt{dt}-s{seconds}-a{action}']=u;r.update(action=action,dt=dt,seconds=seconds,coordinate_rms_change_mm=float(1000*np.sqrt(np.mean((u-baseline['predicted'][action])**2))));rows.append(r);write(OUT/'mpm-fitted-numerical-audit.json',dict(rows=rows,scope='Fixed development fitted parameters; compare production 4 seconds / 0.1 milliseconds against doubled duration or halved timestep. Different material laws are not numerical convergence errors.'));np.savez_compressed(BASE/'mpm-fitted-audit.npz',**fields);print('FITTED_MPM_AUDIT',r,flush=True)
 del model
