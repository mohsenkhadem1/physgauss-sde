"""Isolate a difficult development Newton step and preserve residual evidence."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.material_field import MaterialPositionFixture
from scripts.position_ood.study import BASE,OUT,protocol
from scripts.position_ood.fixture import controls
from scripts.position_ood.pilot import ACTIONS
from scripts.phystwin.consistent_fixture import ConsistentFixture
from scripts.phystwin.sequential_reference import write
m=MaterialPositionFixture(14,**protocol()['development'][3]);m.progress=True;m.set_controls(controls(*ACTIONS[7]));u=np.load(BASE/'references/development-3/checkpoints/n14-q3-6.npz')['full_displacement'];s=m.evaluate(u);print('INITIAL',s['potential_J'],np.linalg.norm(s['residual']),s['J'].min(),flush=True)
try:
 s=ConsistentFixture.solve(m,initial=u,maximum_iterations=6,force_tolerance_N=1e-9);report=dict(converged=True,history=s['history'])
except RuntimeError as e:report=dict(converged=False,error=repr(e),history=getattr(e,'history',[]))
write(OUT/'development-3-newton-diagnosis.json',report);print(json.dumps(report),flush=True)
