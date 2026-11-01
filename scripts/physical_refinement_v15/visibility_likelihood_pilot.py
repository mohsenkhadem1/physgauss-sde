"""Development n6 hidden-material image distinguishability, no new mechanics."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
from scripts.position_ood.fixture import controls
from scripts.physical_refinement_v15.state_scene import scene_from_state
from scripts.physical_refinement_v15.observations import Camera,render_scene,compare_rgbd
from scripts.physical_refinement_v15.feature_likelihood import compare
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=ROOT/'reports/physical-refinement-v15';raw=ROOT/'runs/physical-refinement-v15/visibility-likelihood-pilot';raw.mkdir(parents=True,exist_ok=True)
 sources=[Path(__file__),ROOT/'scripts/physical_refinement_v15/feature_likelihood.py',ROOT/'scripts/physical_refinement_v15/state_scene.py',ROOT/'scripts/physical_refinement_v15/observations.py',ROOT/'scripts/physical_refinement_v15/fixture_geometry.py']
 plan=dict(scope='Development hypotheses/actions only; no noise/camera selection from outcomes; syntheticnoise sensitivity nothardwareclaims',sources={str(p.relative_to(ROOT)):sha(p)for p in sources},views={'left':[-.30,0,0],'oblique':[.12,-.48,.20]},states=['rest','stretch','twist_x'])
 (report/'visibility-likelihood-plan.json').write_text(json.dumps(plan,indent=2)+'\n');images={};evidence={}
 for name in ['gradient_plus','gradient_minus']:
  folder=ROOT/'runs/physical-refinement-v15/mechanics-pilot'/name/'n6-q3';completion=json.loads((folder/'complete.json').read_text());spec=completion['spec']['case'];model=SmoothMaterialFixture(6,**spec,gravity=(0,0,0),field_amplitude_J=0.,order=3,device='cpu')
  for action,i in [('rest',None),('stretch',0),('twist_x',2)]:
   if i is None:u=np.zeros_like(model.rest);poses=np.zeros((2,6))
   else:
    path=folder/f'action{i}.npz';assert sha(path)==completion['artifact_sha256'][path.name];evidence[str(path.relative_to(ROOT))]=sha(path)
    with np.load(path)as data:u=data['nodal_displacement'].copy()
    aname,amp=completion['spec']['actions'][i];assert aname==action;poses=controls(aname,amp)
   meshes=scene_from_state(model,u,spec['layout'],poses)
   for view,eye in plan['views'].items():
    image=render_scene(meshes,Camera(tuple(eye),(.12,0,0),256,192,250.));images[(name,action,view)]=image
    path=raw/f'{name}-{action}-{view}.npz';np.savez_compressed(path,rgb=image['rgb'],depth=image['depth']);evidence[str(path.relative_to(ROOT))]=sha(path)
  del model
 results={}
 for action in plan['states']:
  results[action]={}
  for view in plan['views']:
   a,b=[images[(name,action,view)]for name in ['gradient_plus','gradient_minus']];results[action][view]=dict(pixel_metrics=compare_rgbd(a,b),feature_likelihood=compare(a,b))
 (report/'visibility-likelihood-pilot.json').write_text(json.dumps(dict(plan=plan,results=results,evidence_sha256=evidence),indent=2)+'\n')
 for action in plan['states']:
  for view in plan['views']:print(action,view,[round(x['mahalanobis_squared'],3)for x in results[action][view]['feature_likelihood']['noise_sweep']])
if __name__=='__main__':main()
