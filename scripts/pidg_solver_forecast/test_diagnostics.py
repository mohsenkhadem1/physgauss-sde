import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure
configure(1)
import torch,numpy as np
from scripts.pidg_solver_forecast.geometry import volume
from scripts.pidg_solver_forecast.diagnostics import Interpenetration
rng=np.random.default_rng(19);points=np.concatenate([rng.uniform(-.25,.25,(1000,3))+c for c in [[-.7,0,0],[.7,0,0]]]);g=volume(points,np.ones(len(points)),resolution=12);assert g['components']==2;diag=Interpenetration(g['elements'],g['component_labels']);x=torch.tensor(g['nodes'],device='cuda',dtype=torch.float64);separated=diag(x);assert separated['sampled_interbody_inside_fraction']==0
labels=np.unique(g['component_labels']);body=np.unique(g['elements'][g['component_labels']==labels[1]]);x[body,0]-=1.15;overlap=diag(x);assert overlap['sampled_interbody_inside_fraction']>0 and overlap['sampled_interbody_max_depth_normalized']>0
result=dict(passed=True,separated=separated,overlap=overlap,scope='Known synthetic disjoint and interpenetrating volumes; tests reporting, not collision response.')
(ROOT/'reports/pidg-solver-forecast-v01/diagnostic-tests.json').write_text(json.dumps(result,indent=2));print(result)
