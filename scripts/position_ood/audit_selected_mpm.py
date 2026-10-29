"""Report fixed-method numerical checks at the final development fitted parameters."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,within_deadline
from scripts.position_ood.seal import verify
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.phystwin.sequential_reference import write
seal=verify();directory=BASE/seal['settings']['mpm']['tag']/'development-1'
while not (directory/'summary.json').exists():within_deadline();time.sleep(20)
fit=json.loads((directory/'summary.json').read_text());baseline=np.load(directory/'predictions.npz');rows=[]
for dt,seconds in [(fit['dt'],2*fit['seconds']),(fit['dt']/2,fit['seconds'])]:
 model=PositionMPM(fit['resolution'],fit['grid'],dt,rpic=fit['rpic'],attachment_width=fit['attachment_width'],ramp_seconds=fit['ramp_seconds']);model.parameters(**parameters(fit['best']['z'],fit['best']['layout']))
 for action in [0,4,6]:
  within_deadline();r=model.solve(baseline['poses'][action],seconds=seconds);u=r.pop('field');r.update(action=action,dt=dt,seconds=seconds,coordinate_rms_change_mm=float(1000*np.sqrt(np.mean((u-baseline['predicted'][action])**2))));rows.append(r);write(OUT/'mpm-selected-numerical-audit.json',dict(tag=seal['settings']['mpm']['tag'],rows=rows,scope='Post-freeze development numerical diagnostic at fitted parameters. Reports approximation error; it does not tune frozen settings or read held-out test scores.'));print('SELECTED_MPM_AUDIT',r,flush=True)
 del model
