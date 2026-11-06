"""Calibration RGB gradients through actual dynamics; forecasts use frozen parameters.

Direct inverse dynamics supplies initialization. This optional calibration stage
does optimize rollouts; it must not be described as iteration-free training.
"""
import argparse,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--steps',type=int,default=80);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from PIL import Image
 from torch.utils.checkpoint import checkpoint
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
 from utils.loss_utils import ssim
 torch.set_num_threads(2);torch.manual_seed(19);rng=np.random.default_rng(19)
 fit=json.loads(Path(a.fit).read_text());assert fit['completed'] and not fit['future_positions_read'];kind=fit['kind'];renderer=MechanicalRenderer();m=renderer.model
 base=ROOT/'runs/solver-driven-gaussians-v02';out=Path(a.output);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((base/'images/manifest.json').read_text())
 records=[r for r in manifest['records'] if r['split'] in ['fit','validation']]
 for r in records:
  r['target']=torch.tensor(np.array(Image.open(base/'images'/r['image']))/255.,device='cuda',dtype=torch.float32).permute(2,0,1);r['mask']=(r['target']-.96).abs().max(0).values>.03
 validation=[r for r in records if r['split']=='validation'];training=[r for r in records if r['split']=='fit' and r['frame'] in [20,40,60,70]]
 raw=torch.nn.Parameter(torch.log(m.t(fit['parameters']).clamp_min(.1)));optimizer=torch.optim.Adam([raw],lr=.025)
 lower=m.t([.15]+[.01]*8+[1.,.01]).log();upper=m.t([6.]+[2000.]*8+[1000.,20.]).log();dt=.0002
 with torch.no_grad():raw.clamp_(lower,upper)
 def simulate(theta,steps,ad=False,save=False):
  x=m.rest.clone();v=torch.zeros_like(x);states=[x]
  for begin in range(0,steps,20):
   end=min(steps,begin+20)
   def segment(xx,vv,pp,begin=begin,end=end):
    for j in range(begin,end):
     f,_,_=m.force(xx,vv,j*dt,pp,'calibration',kind,create_graph=ad);vv=vv+dt*f/m.mass[:,None];xx=xx+dt*vv
    return xx,vv
   if ad:x,v=checkpoint(segment,x,v,theta,use_reentrant=False)
   else:
    # Validation retains exact 2-ms states for the declared reserved frames.
    if save:
     for j in range(begin,end):
      f,_,_=m.force(x,v,j*dt,theta,'calibration',kind);v=v+dt*f/m.mass[:,None];x=x+dt*v
      if (j+1)%10==0:states.append(x)
    else:x,v=segment(x,v,theta)
  return torch.stack(states) if save else x
 best=None;history=[];start=time.time();last_valid=raw.detach().clone();failures=[]
 for iteration in range(a.steps+1):
  if iteration%10==0 or iteration==a.steps:
   with torch.no_grad():
    trajectory=simulate(raw.exp(),700,save=True);scores=[]
    for r in validation:
     pred=renderer.image(trajectory[r['frame']],r['view']);scores.append(float(-10*torch.log10((pred-r['target']).square()[:,r['mask']].mean().clamp_min(1e-12))))
   score=float(np.mean(scores));history.append(dict(iteration=iteration,validation_foreground_psnr=score,seconds=time.time()-start));print(history[-1],flush=True)
   if best is None or score>best['score']:
    best=dict(score=score,iteration=iteration,parameters=raw.exp().detach().cpu().tolist())
   (out/'status.json').write_text(json.dumps(dict(completed=False,best=best,history=history,failures=failures),indent=2))
  if iteration==a.steps:break
  # Progressively increase the observed simulation horizon; no future target use.
  allowed=[r for r in training if r['time_s']<=(.041 if iteration<10 else .081 if iteration<20 else .141)]
  r=allowed[int(rng.integers(len(allowed)))];optimizer.zero_grad(set_to_none=True)
  try:
   x=simulate(raw.exp(),round(r['time_s']/dt),ad=True);pred=renderer.image(x,r['view']);loss=.4*(pred-r['target']).abs().mean()+.4*(pred-r['target']).abs()[:,r['mask']].mean()+.2*(1-ssim(pred,r['target']))
   loss.backward()
   if not torch.isfinite(raw.grad).all():raise ValueError('Nonfinite calibration gradient')
   torch.nn.utils.clip_grad_norm_([raw],5.);last_valid=raw.detach().clone();optimizer.step()
   with torch.no_grad():raw.clamp_(lower,upper)
  except (ValueError,RuntimeError) as error:
   failures.append(dict(iteration=iteration,error=str(error)));raw.data.copy_(last_valid)
   for group in optimizer.param_groups:group['lr']*=.5
 result=dict(completed=True,kind=kind,parameters=best['parameters'],best=best,history=history,failures=failures,calibration_end_s=.14,future_positions_read=False,future_images_read=False,force_labels_read=False,material_labels_read=False,initial_fit_sha256=hashlib.sha256(Path(a.fit).read_bytes()).hexdigest(),scope='Known geometry/density/cameras. Material and Gaussian attachment/load coefficients fitted through actual dynamic RGB rendering; appearance fixed. Direct-ID initialization plus iterative calibration, not iteration-free training. Final forecasts perform forward integration only.')
 (out/'fit.json').write_text(json.dumps(result,indent=2));print(best,flush=True)
if __name__=='__main__':main()
