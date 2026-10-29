"""Development check: smooth positional control versus impulsive endpoints."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_reference import write
obs=np.load(BASE/'observations/development-1.npz');ref=np.load(BASE/'fields/development-1.npz')['field'];rows=[]
for width in [0.,.006]:
 p=BASE/('mpm' if width==0 else 'mpm-layer0.006')/'development-1/summary.json';r=json.loads(p.read_text());m=PositionMPM(2,64,.0001,rpic=-1.,attachment_width=width,ramp_seconds=1.);m.parameters(**parameters(r['best']['z'],r['best']['layout']))
 for action in [0,4,7,8]:
  within_deadline()
  try:
   s=m.solve(obs['poses'][action],seconds=4.,block=20);u=s.pop('field');s.update(action=action,width=width,rmse_mm=float(1000*np.sqrt(np.mean((u-ref[action])**2))),numerically_valid=True,settled=bool(s['drift_m']<2e-5))
  except RuntimeError as e:s=dict(action=action,width=width,numerically_valid=False,settled=False,error=repr(e))
  rows.append(s);write(BASE/'mpm-ramp-audit.json',rows);print('RAMP_AUDIT',s,flush=True)
 del m
