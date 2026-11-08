"""Fit calibration RGB only; compare fixed-appearance visual motion adapters.

PIDG here means the released deformation network/TV model, not a claim of an
unmodified full PIDG training recipe. Mechanics inference remains a later stage.
"""
import argparse,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--method',choices=['nodes','pidg','hybrid'],default='nodes');p.add_argument('--steps',type=int,default=2500);p.add_argument('--initialize');a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import numpy as np,torch
 from PIL import Image
 from scipy.interpolate import BSpline
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
 from utils.loss_utils import ssim
 torch.set_num_threads(2);torch.manual_seed(0);rng=np.random.default_rng(7)
 renderer=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v01';folder=base/'images';doc=json.loads((folder/'manifest.json').read_text());out=base/('rgb-'+a.method);out.mkdir(exist_ok=True)
 records=[r for r in doc['records'] if r['split'] in ['fit','validation']]
 for r in records:
  r['target']=torch.tensor(np.array(Image.open(folder/r['image']))/255.,device='cuda',dtype=torch.float32).permute(2,0,1);r['mask']=(r['target']-.96).abs().max(0).values>.03
 fit=[r for r in records if r['split']=='fit'];validation=[r for r in records if r['split']=='validation']
 times=np.arange(71)*.002;knots=np.r_[np.zeros(4),np.linspace(0,1,10)[1:-1],np.ones(4)];B=torch.tensor(BSpline(knots,np.eye(12),3)(times/.14),device='cuda',dtype=torch.float32)
 prior=None
 if a.method=='nodes':
  controls=torch.nn.Parameter(torch.zeros(10,len(renderer.model.rest),3,device='cuda'))
  if a.initialize:
   initial=np.load(a.initialize)['x']-renderer.model.rest.cpu().numpy();coeff=np.linalg.lstsq(B[:,2:].cpu().numpy(),initial.reshape(len(times),-1),rcond=None)[0].reshape(10,len(renderer.model.rest),3);controls.data.copy_(torch.tensor(coeff,device='cuda'))
  optimizer=torch.optim.Adam([controls],lr=.00008 if a.initialize else .0003)
  edges=np.unique(np.sort(np.concatenate([renderer.model.elements.cpu().numpy()[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0);edges=torch.tensor(edges,device='cuda');length=(renderer.model.rest[edges[:,0]]-renderer.model.rest[edges[:,1]]).norm(dim=-1)
  def positions(frame):return renderer.model.rest+torch.einsum('k,kni->ni',B[frame,2:],controls)
  def prediction(r):return renderer.image(positions(r['frame']),r['view'])
  def state():return {'controls':controls.detach().cpu().clone()}
 else:
  from scene_PIDG import DeformModel
  from gaussian_renderer import render as native_render
  from types import SimpleNamespace
  from utils.general_utils import quaternion_to_matrix
  cfg={};exec((EXTERNAL/'upstream/arguments/pidg/rubber_dack.py').read_text(),cfg)
  model=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=True,reg_temporal_able=True)
  for head in [model.deform.gaussian_warp,model.deform.gaussian_rotation,model.deform.gaussian_scaling]:torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
  optimizer=torch.optim.Adam([{'params':model.grid4d.parameters(),'lr':.005},{'params':model.deform.parameters(),'lr':.0008}])
  class MeterDeform:
   def step(self,x,t):
    d=model.step(x/.1,t);d['d_xyz']=torch.cat([d['d_xyz'][:,:4],d['d_xyz'][:,4:]*.1],-1);return d
  meter=MeterDeform();pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
  if a.method=='hybrid':
   from scripts.pidg_dense_mechanics.field import DenseMechanics
   with torch.random.fork_rng(devices=[torch.cuda.current_device()]):prior=DenseMechanics()
  def packet(t,x=None):
   x=renderer.appearance.get_xyz if x is None else x
   return meter.step(x,torch.full((len(x),1),t/.42,device='cuda'))
  def prediction(r):
   d=packet(r['time_s']);return native_render(renderer.cameras[r['view']],renderer.appearance,pipe,renderer.background,d['d_xyz'],d['d_rotation'],d['d_scaling'],False)['render'].clamp(0,1)
  def positions(frame):
   x=renderer.model.rest;d=packet(times[frame],x)['d_xyz'];return (quaternion_to_matrix(d[:,:4])@x[...,None]).squeeze(-1)+d[:,4:]
  def state():return {'grid':{k:v.detach().cpu().clone() for k,v in model.grid4d.state_dict().items()},'network':{k:v.detach().cpu().clone() for k,v in model.deform.state_dict().items()}}
 best=None;history=[];start=time.time()
 for step in range(a.steps+1):
  if step%250==0 or step==a.steps:
   with torch.no_grad():
    values=[]
    for r in validation:
     pred=prediction(r);mse=(pred-r['target']).square()[:,r['mask']].mean();values.append(float(-10*torch.log10(mse.clamp_min(1e-12))))
   score=float(np.mean(values));history.append(dict(step=step,validation_foreground_psnr=score,seconds=time.time()-start))
   if best is None or score>best['score']:
    best=dict(score=score,step=step);torch.save(dict(**state(),method=a.method,step=step),out/'best.pt')
   (out/'status.json').write_text(json.dumps(dict(completed=False,history=history,best=best),indent=2));print(a.method,history[-1],flush=True)
  if step==a.steps:break
  r=fit[int(rng.integers(len(fit)))];optimizer.zero_grad(set_to_none=True);pred=prediction(r)
  loss=.4*(pred-r['target']).abs().mean()+.4*(pred-r['target']).abs()[:,r['mask']].mean()+.2*(1-ssim(pred,r['target']))
  if a.method=='nodes':
   u=positions(r['frame'])-renderer.model.rest;F=torch.eye(3,device='cuda')+torch.einsum('eai,eaj->eij',u[renderer.model.elements],renderer.model.grad)
   z=F.new_tensor([0.,0.,-1.]).expand(len(F),3)
   elastic=renderer.model.law.energy(F,z).mean()/12000.
   loss=loss+1e-3*elastic+2e-3*((u[edges[:,0]]-u[edges[:,1]])/length[:,None]).square().mean()+1e-4*((controls[2:]-2*controls[1:-1]+controls[:-2])/.01).square().mean()
  else:
   d=packet(r['time_s']);loss=loss+.5*d['reg_spatial'].mean()+.5*d['reg_temporal'].mean()
   if prior is not None:loss=loss+1e-5*prior(meter,renderer.appearance,torch.tensor(r['time_s']/.42,device='cuda'))
  loss.backward()
  if prior is not None:prior.step(1e-5)
  before=controls.detach().clone() if a.method=='nodes' else None
  optimizer.step()
  if a.method=='nodes':
   with torch.no_grad():
    delta=controls-before
    for backtrack in range(16):
     all_u=torch.einsum('tk,kni->tni',B[:,2:],controls)
     all_F=torch.eye(3,device='cuda')+torch.einsum('teai,eaj->teij',all_u[:,renderer.model.elements],renderer.model.grad)
     if bool((torch.linalg.det(all_F)>.2).all()):break
     controls.copy_(before+delta*2**(-backtrack-1))
    else:controls.copy_(before)
  if step==int(a.steps*.65):
   for group in optimizer.param_groups:group['lr']*=.2
 checkpoint=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
 if a.method=='nodes':controls.data.copy_(checkpoint['controls'])
 else:model.grid4d.load_state_dict(checkpoint['grid']);model.deform.load_state_dict(checkpoint['network'])
 with torch.no_grad():x=torch.stack([positions(i) for i in range(len(times))]).cpu().numpy()
 np.savez_compressed(out/'observed-positions.npz',x=x,times=times,rest=renderer.model.rest.cpu().numpy())
 if prior is not None:prior.save(out/'last-mechanics.pt')
 result=dict(completed=True,method=a.method,best=best,history=history,steps=a.steps,future_images_read=False,reference_nodes_read=False,appearance_frozen=True,gaussians=len(renderer.cov),checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),scope='Controlled fixed-appearance RGB motion adapters; PIDG uses author deformation/TV architecture, not unmodified full PIDG momentum/flow training. Known block geometry/cameras. Inferred trajectories precede material/attachment identification.')
 (out/'complete.json').write_text(json.dumps(result,indent=2));print(result['best'],flush=True)
if __name__=='__main__':main()
