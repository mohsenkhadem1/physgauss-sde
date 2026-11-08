"""Create calibrated RGB observations and verify appearance conversion identity."""
import argparse,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import numpy as np,torch
 from PIL import Image
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer,APPEARANCE
 from scripts.mechanics_gaussian_bridge.bridge import render
 torch.set_num_threads(2);renderer=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v01';out=base/'images';out.mkdir(exist_ok=True)
 states=np.load(base/'reference/calibration.npz');records=[];identity=[]
 with torch.no_grad():
  explicit=dict(centers=renderer.appearance.get_xyz,covariance=renderer.cov,rotation=torch.eye(3,device='cuda').expand(len(renderer.cov),3,3))
  for view in range(4):
   old=render(renderer.cameras[view],renderer.appearance,explicit,renderer.background)['render'];new=renderer.image(renderer.model.rest,view);identity.append(float((old.clamp(0,1)-new).abs().max()))
  assert max(identity)<1e-4,identity
  # Only observed prefix is exported to this fitting dataset; no future images.
  for frame,t in enumerate(states['times']):
   if t>.14000001:break
   if frame%2:continue
   for view in range(4):
    rgb=renderer.image(states['x'][frame],view).permute(1,2,0).cpu().numpy();name=f'{frame:04d}-view{view}.png';Image.fromarray((rgb*255).round().astype('uint8')).save(out/name)
    split='novel_view' if view>=2 else ('validation' if frame%10==6 else 'fit')
    records.append(dict(frame=frame,time_s=float(t),view=view,split=split,image=name))
 (out/'manifest.json').write_text(json.dumps(dict(records=records,scope='Known initial appearance and block geometry; RGB-only trajectory reconstruction receives images/cameras, not generator nodes.',appearance_sha256=hashlib.sha256(APPEARANCE.read_bytes()).hexdigest()),indent=2))
 report=dict(passed=True,initial_rgb_max_difference=max(identity),per_view=identity,gaussians=len(renderer.cov),extrapolated_appearance_fraction=renderer.binding.extrapolated_fraction,images=len(records))
 (ROOT/'reports/solver-driven-gaussians-v01/appearance-binding-audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
