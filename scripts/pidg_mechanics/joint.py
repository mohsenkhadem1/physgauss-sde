"""Image -> mechanics adjoint -> inferred boundary/material -> renderer loop."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(int(os.environ.get('PIDG_GPU','0')));sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_mechanics.appearance import Appearance,camera_frame
from scripts.mechanics_gaussian_bridge.implicit_boundary import BoundaryEquilibrium
from scripts.mechanics_gaussian_bridge.bridge import render

class Joint(torch.nn.Module):
 def __init__(self,appearance):
  super().__init__();self.appearance=appearance;self.controls=torch.nn.Parameter(torch.zeros(12,6,4,device='cuda',dtype=torch.double));self.material=torch.nn.Parameter(torch.zeros(6,3,device='cuda',dtype=torch.double))
 def values(self,frame,b):
  p=frame/149*11;i=min(int(p),10);delta=(1-(p-i))*self.controls[i,b]+(p-i)*self.controls[i+1,b]
  m=self.appearance.models[b];theta=self.appearance.base_parameters[b].clone();theta[1:7]=theta[1:7]+.7*self.material[b,0].tanh();theta[m.extra]=theta[m.extra]+.5*self.material[b,1].tanh();theta[m.extra+2]=theta[m.extra+2]+.5*self.material[b,2].tanh()
  target=m.targets[frame];center=target.mean((0,1),keepdim=True);target=center+(target-center)*torch.exp(.25*delta[3].tanh())+.3*delta[:3].tanh()[None,None]
  return theta,target
 def state(self,frame):
  states=[];a=self.appearance
  for b,(m,binding) in enumerate(zip(a.models,a.bindings)):
   theta,target=self.values(frame,b);targets=m.targets.clone();targets[frame]=target
   initial=target.mean(0).detach();nodes=BoundaryEquilibrium.apply(theta,targets,m,frame,initial)
   state=binding.transport(nodes);F=torch.einsum('pai,paJ->piJ',nodes[binding.ids],binding.grad)
   state['centers']=state['centers']+.25*a.radius_raw[b].tanh()*torch.einsum('pij,pj->pi',F,a.reference_radials[b]);state['covariance']=state['covariance']*torch.exp(.6*a.scale_raw[b].tanh())
   states.append(state)
  return {k:torch.cat([s[k] for s in states]).float() for k in ['centers','covariance','rotation']}
 def image(self,camera,frame):return render(camera,self.appearance,self.state(frame),torch.zeros(3,device='cuda'))['render']

def main():
 torch.set_num_threads(1);torch.manual_seed(43);root=ROOT/os.environ.get('PIDG_MECHANICS_ROOT','runs/pidg-mechanics-v02');a=Appearance();checkpoint=torch.load(root/'appearance-best.pt',weights_only=False)
 with torch.no_grad():
  for k in ['features','radius_raw','scale_raw','opacity_raw']:getattr(a,k).copy_(checkpoint[k])
 j=Joint(a)
 if os.environ.get('PIDG_JOINT_RESUME')=='1':
  saved=torch.load(root/'joint-best.pt',weights_only=False)
  with torch.no_grad():
   for k,value in saved['appearance'].items():getattr(a,k).copy_(value)
   j.controls.copy_(saved['controls']);j.material.copy_(saved['material'])
 opt=torch.optim.Adam([{'params':a.parameters(),'lr':.003},{'params':[j.controls],'lr':.015},{'params':[j.material],'lr':.02}]);base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit';tr=json.loads((base/'transforms_train.json').read_text());va=json.loads((base/'transforms_test.json').read_text());training=[camera_frame(f,tr) for f in tr['frames']];validation=[camera_frame(f,va) for f in va['frames']];best=-float('inf');history=[];start=time.time()
 steps=int(os.environ.get('PIDG_JOINT_STEPS','80'));every=int(os.environ.get('PIDG_JOINT_EVERY','20'));gradient_records=[]
 for step in range(steps+1):
  if step:
   camera,gt,mask,frame=training[(step*37)%len(training)];pred=j.image(camera,frame)
   loss=(pred[:,mask]-gt[:,mask]).square().mean()+.2*(pred-gt).abs().mean()+.0001*j.material.square().mean()+.0001*j.controls.square().mean()+.001*(j.controls[1:]-j.controls[:-1]).square().mean()
   opt.zero_grad();loss.backward();assert j.controls.grad is not None and j.material.grad is not None;assert all(p.grad is None or torch.isfinite(p.grad).all() for p in j.parameters());gradient_records.append({'step':step,'boundary_gradient_norm':float(j.controls.grad.norm()),'material_gradient_norm':float(j.material.grad.norm())});opt.step()
  if step%every==0 or step==steps:
   values=[]
   # Solver uses enable_grad internally to obtain constitutive derivatives.
   for camera,gt,mask,frame in validation:
    pred=j.image(camera,frame).detach().clamp(0,1);e=(pred-gt).square();values.append([float(-10*e.mean().log10()),float(-10*e[:,mask].mean().log10())])
   mean=np.mean(values,0);record={'step':step,'validation_psnr':float(mean[0]),'validation_foreground_psnr':float(mean[1]),'seconds':time.time()-start};history.append(record)
   if mean[1]>best:
    best=mean[1];torch.save({'appearance':{k:getattr(a,k).detach().clone() for k in ['features','radius_raw','scale_raw','opacity_raw']},'controls':j.controls.detach().clone(),'material':j.material.detach().clone(),'step':step},root/'joint-best.pt')
   (root/'joint-progress.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
 (root/'joint-selection.json').write_text(json.dumps({'completed':True,'test_images_read':False,'criterion':'Validation foreground PSNR','best_foreground_psnr':best,'steps':steps,'warm_restart':os.environ.get('PIDG_JOINT_RESUME')=='1','gradient_records':gradient_records,'scope':'Image gradients through converged equilibrium, frozen learned material decoder, inferred controls, SH appearance. No reference novel-view images.'},indent=2))
if __name__=='__main__':main()
