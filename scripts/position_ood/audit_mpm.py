"""Development-only resolution and settling diagnostic for the MPM adapter."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.study import BASE,within_deadline
from scripts.phystwin.sequential_reference import write
p=BASE/'mpm/development-1/summary.json';r=json.loads(p.read_text());obs=np.load(BASE/'observations/development-1.npz');ref=np.load(BASE/'fields/development-1.npz')['field'];out=BASE/'mpm/development-1/audit-pic.json';rows=[]
for resolution,grid,seconds in [(3,96,2.),(3,96,6.)]:
 m=PositionMPM(resolution,grid,.0001,rpic=-1.);m.parameters(**parameters(r['best']['z'],r['best']['layout']))
 for action in [0,4,7,8]:
  within_deadline();s=m.solve(obs['poses'][action],seconds=seconds);field=s.pop('field');s.update(resolution=resolution,grid=grid,seconds=seconds,action=action,rmse_mm=float(1000*np.sqrt(np.mean((field-ref[action])**2))));rows.append(s);write(out,rows);print('AUDIT',s,flush=True)
 del m
