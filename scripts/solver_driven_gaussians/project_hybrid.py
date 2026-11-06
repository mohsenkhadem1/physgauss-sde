"""Extend fitted dense surface motion into a volume using affine-exact smoothing."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from scipy import sparse
 from scipy.spatial import cKDTree
 from scipy.sparse.linalg import splu
 from PIL import Image
 from scene_PIDG import DeformModel
 from utils.general_utils import quaternion_to_matrix
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
 torch.set_num_threads(2);renderer=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v02';out=base/'hybrid-projection';out.mkdir(exist_ok=True)
 cfg={};exec((EXTERNAL/'upstream/arguments/pidg/rubber_dack.py').read_text(),cfg);d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False)
 checkpoint=torch.load(base/'rgb-hybrid/best.pt',map_location='cuda',weights_only=False);d.grid4d.load_state_dict(checkpoint['grid']);d.deform.load_state_dict(checkpoint['network'])
 q=renderer.appearance.get_xyz;points=q.cpu().numpy();rest=renderer.model.rest.cpu().numpy();N=renderer.binding.N.cpu().numpy();ids=renderer.binding.ids.cpu().numpy();count=len(rest)
 B=sparse.coo_matrix((N.ravel(),(np.repeat(np.arange(len(q)),4),ids.ravel())),shape=(len(q),count)).tocsr()
 near=cKDTree(rest).query(rest,k=17)[1][:,1:];lr=[];lc=[];lv=[]
 for i,nb in enumerate(near):
  Q=np.c_[np.ones(len(nb)),rest[nb]-rest[i]];weights=np.linalg.pinv(Q)[0]
  lr.extend([i]*(len(nb)+1));lc.extend([i,*nb]);lv.extend([1.,*(-weights)])
 L=sparse.coo_matrix((lv,(lr,lc)),shape=(count,count)).tocsr();W=renderer.appearance.get_opacity[:,0].cpu().numpy()
 times=np.arange(71)*.002;dense=[]
 with torch.no_grad():
  for t in times:
   packet=d.step(q/.1,torch.full((len(q),1),t/.42,device='cuda'))['d_xyz'];x=(quaternion_to_matrix(packet[:,:4])@(q/.1)[...,None]).squeeze(-1)+packet[:,4:];dense.append(x.cpu().numpy()*.1)
 dense=np.asarray(dense);doc=json.loads((base/'images/manifest.json').read_text());validation=[r for r in doc['records'] if r['split']=='validation'];records=[];best=None
 for strength in [1.,10.,100.]:
  factor=splu((B.T@sparse.diags(W)@B+strength*(L.T@L)+sparse.eye(count)*1e-8).tocsc());states=[];alphas=[]
  for target in dense:
   x=rest+factor.solve(B.T@(W[:,None]*(target-points)))
   # Keep a proper rigid component; reduce only residual deformation if needed.
   c=points.mean(0);cc=target.mean(0);U,_,Vt=np.linalg.svd((points-c).T@(target-cc));R=Vt.T@U.T
   if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
   rigid=(rest-c)@R.T+cc;alpha=1.
   while float(torch.linalg.det(renderer.model.deformation(renderer.model.t(x))).min())<.3:
    alpha*=.5;x=rigid+alpha*(rest+factor.solve(B.T@(W[:,None]*(target-points)))-rigid)
   states.append(x);alphas.append(alpha)
  states=np.asarray(states);states[0]=rest
  with torch.no_grad():
   scores=[]
   for r in validation:
    target=torch.tensor(np.array(Image.open(base/'images'/r['image']))/255.,device='cuda',dtype=torch.float32).permute(2,0,1);mask=(target-.96).abs().max(0).values>.03;pred=renderer.image(states[r['frame']],r['view']);scores.append(float(-10*torch.log10((pred-target).square()[:,mask].mean().clamp_min(1e-12))))
  record=dict(strength=strength,validation_foreground_psnr=float(np.mean(scores)),minimum_deformation_fraction=min(alphas));records.append(record)
  if best is None or record['validation_foreground_psnr']>best['validation_foreground_psnr']:
   best=record;np.savez_compressed(out/'observed-positions.npz',x=states,times=times,rest=rest)
 (out/'selection.json').write_text(json.dumps(dict(candidates=records,selected=best,reference_nodes_read=False,future_images_read=False),indent=2));print(records)
if __name__=='__main__':main()
