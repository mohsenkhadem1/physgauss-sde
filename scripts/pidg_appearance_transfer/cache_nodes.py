import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.update(PIDG_MECHANICS_ROOT='runs/pidg-mechanics-v04',PIDG_SH_DEGREE='3',PIDG_GPU='1')
from scripts.pidg_mechanics.joint import Joint
from scripts.pidg_mechanics.appearance import Appearance
import torch,numpy as np
from scripts.mechanics_gaussian_bridge.implicit_boundary import BoundaryEquilibrium
torch.set_num_threads(1);a=Appearance();j=Joint(a);old=ROOT/'runs/pidg-mechanics-v04';saved=torch.load(old/'joint-best.pt',weights_only=False)
with torch.no_grad():j.controls.copy_(saved['controls']);j.material.copy_(saved['material'])
output=ROOT/'runs/pidg-appearance-transfer-v01';nodes=[];targets=[];theta=[]
for b,m in enumerate(a.models):
 body=[];controls=[]
 for frame in range(150):
  th,target=j.values(frame,b);oldtargets=m.targets;m.targets=oldtargets.clone();m.targets[frame]=target.detach()
  x,H,diag=m.solve(th.detach(),frame,target.mean(0).detach());body.append(x.cpu().numpy());controls.append(target.detach().cpu().numpy());m.targets=oldtargets
 nodes.append(body);targets.append(controls);theta.append(th.detach().cpu().numpy());print('body',b,flush=True)
np.savez_compressed(output/'physical-cache.npz',nodes=np.stack(nodes,1),targets=np.stack(targets,1),theta=theta)
