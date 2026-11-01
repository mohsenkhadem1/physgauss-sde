"""Development equilibrium continuation check; not dynamic time convergence."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
from scripts.position_ood.fixture import controls
from scripts.position_ood.reference_solver import solve
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,CASES,write,sha
from scripts.physical_refinement_v15.mechanics_qualification import wrench
model=SmoothMaterialFixture(4,**CASES['homogeneous'],gravity=(0,0,0),device='cuda:0');target=controls('twist_x');state=None
for fraction in (.5,1.):
 model.set_controls(target*fraction);state=solve(model,None if state is None else state['displacement_local_m'],force_tolerance_N=1e-9)
sample=model.sampled(state);f,t=wrench(model,state['displacement_local_m'],target);original=np.load(BASE/'mechanics-pilot/homogeneous/n4-q3/action2.npz');fo,to=wrench(model,original['nodal_displacement'],target)
r=dict(scope='Development only; same quasi-static final equilibrium reached with two increments vs prior warm-started single jump; not time integration',source_sha256=sha(Path(__file__)),field_rms_difference_mm=float(np.sqrt(np.mean((sample['field']-original['field'])**2))*1000),force_max_difference_N=float(np.max(np.linalg.norm(f-fo,axis=1))),torque_max_difference_Nm=float(np.max(np.linalg.norm(t-to,axis=1))),residual_N=sample['residual_N'],minimum_J=sample['minimum_J'])
np.savez_compressed(BASE/'mechanics-path-check.npz',**sample);write(REPORT/'mechanics-path-check.json',r);print(r)
