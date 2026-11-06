"""Kinematic sanity of sealed temporal fields; no true physical labels implied."""
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_solver_forecast.common import BASE,DATA,load
from scripts.pidg_comparison.mechanics import deformation_gradient
from utils.general_utils import quaternion_to_matrix
torch.set_num_threads(2);base=BASE/a.scene;seal=json.loads((base/'evaluation/selection.json').read_text());frames=json.loads((DATA/a.scene/'forecast_test.json').read_text())['frames'];t0=json.loads((DATA/a.scene/'fit.json').read_text())['frames'][0]['time'];rows=[]
for method,arm in [('native-refined',seal['native_selected']),('hybrid','hybrid')]:
 g,d=load(a.scene,arm);active=torch.where(g.get_opacity[:,0]>.1)[0];ids=active[torch.linspace(0,len(active)-1,min(256,len(active)),device='cuda').long()];q=g.get_xyz[ids];length=float((q.max(0).values-q.min(0).values).max())
 def mapping(q,t):
  delta=d.step(q,torch.full((len(q),1),t,device='cuda'))['d_xyz'];return (quaternion_to_matrix(delta[:,:4])@q[...,None]).squeeze(-1)+delta[:,4:]
 with torch.no_grad():
  for fraction in [.0025,.005]:
   F0=deformation_gradient(lambda x:mapping(x,t0),q,epsilon=fraction*length);J0=torch.linalg.det(F0);valid=J0>.05
   for f in frames:
    F=deformation_gradient(lambda x:mapping(x,f['time']),q,epsilon=fraction*length);J=torch.linalg.det(F);relative=F[valid]@torch.linalg.inv(F0[valid]);jr=torch.linalg.det(relative);rows.append(dict(method=method,frame=Path(f['file_path']).name,epsilon_fraction=fraction,sampled_points=len(q),canonical_inversion_fraction=float((J<=0).float().mean()),initial_invalid_fraction=float((~valid).float().mean()),relative_inversion_fraction=float((jr<=0).float().mean()) if len(jr) else None,relative_mean_abs_volume_change=float((jr-1).abs().mean()) if len(jr) else None))
 del g,d;torch.cuda.empty_cache()
summary=[]
for method in ['native-refined','hybrid']:
 for fraction in [.0025,.005]:
  rr=[r for r in rows if r['method']==method and r['epsilon_fraction']==fraction];summary.append(dict(method=method,epsilon_fraction=fraction,**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in ['canonical_inversion_fraction','initial_invalid_fraction','relative_inversion_fraction','relative_mean_abs_volume_change']}))
report=dict(scene=a.scene,summary=summary,rows=rows,test_images_read=False,scope='Finite-difference local Jacobians of each temporal field at 256 deterministic high-opacity canonical points, two step sizes. Relative values use positive, nonsingular first-frame maps only; excluded initial points reported. Kinematic sanity, not material accuracy or ground-truth dynamics. Solver tetrahedral diagnostics use a different discretization and are not a common physical-error leaderboard.')
(ROOT/'reports/pidg-solver-forecast-v01'/a.scene/'temporal-physics.json').write_text(json.dumps(report,indent=2));print('Physics audit complete',a.scene)
