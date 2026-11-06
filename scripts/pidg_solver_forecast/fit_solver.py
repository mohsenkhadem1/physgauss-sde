"""Multiple-shooting RGB calibration; sealed forward-only suffix predictions."""
import sys,argparse,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--steps',type=int,default=300);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from utils.loss_utils import ssim
 from scripts.pidg_solver_forecast.common import BASE,DATA,camera
 from scripts.pidg_solver_forecast.model import SceneFEM
 torch.set_num_threads(2);torch.manual_seed(31);rng=np.random.default_rng(31);out=BASE/a.scene/'solver';adapter=np.load(out/'adapter.npz');appearance=torch.load(out/'appearance.pt',map_location='cpu',weights_only=False);m=SceneFEM(adapter,appearance);m.set_binding_tets();m.initialize_loads();m.clamp();optimizer=torch.optim.Adam([{'params':[m.param['log_scale']],'lr':.01},{'params':[m.param['log_stiffness'],m.param['log_damping']],'lr':.03},{'params':[m.param['loads']],'lr':.2},{'params':[m.param['body']],'lr':.1}])
 training=json.loads((DATA/a.scene/'fit.json').read_text())['frames'];validation=json.loads((DATA/a.scene/'forecast_validation.json').read_text())['frames'];train=[(f,*camera(a.scene,f)) for f in training];val=[(f,*camera(a.scene,f)) for f in validation];best=None;history=[];failures=[];start=time.time();modes=['constant','decay','zero','linear']
 def validate(mode,terminal='projected'):
  x=m.observed[-1].clone();v=m.observed_v[-1].clone();t=m.end;scores=[]
  if terminal=='integrated_tail':
   idx=max(0,len(m.times)-11);x,v=m.integrate(m.observed[idx].clone(),m.observed_v[idx].clone(),float(m.times[idx]),m.end)
  for f,cam,gt,mask in val:
   x,v=m.integrate(x,v,t,f['time'],mode);t=f['time'];pred=m.image(x,cam);scores.append(float(-10*(pred-gt).square()[:,mask].mean().clamp_min(1e-12).log10()))
  return float(np.mean(scores))
 # Preserve a stable unforced control initialization if derivative-based forces are poor.
 initial={k:v.detach().clone() for k,v in m.param.items()};initial_candidates=[];best_initial=None
 for velocity_factor in [0.,.25,1.]:
  m.set_velocity_factor(velocity_factor)
  for material_scale in [.03,.3,1.]:
   for multiplier in [0.,.1,1.]:
    with torch.no_grad():
     for k,v in initial.items():m.param[k].copy_(v)
     m.param['log_scale'].fill_(float(np.log(material_scale)));m.param['loads'].mul_(multiplier);m.param['body'].mul_(multiplier)
     try:score=validate('constant')
     except (RuntimeError,ValueError):score=-1e9
     initial_candidates.append(dict(velocity_factor=velocity_factor,material_scale=material_scale,load_multiplier=multiplier,validation_foreground_psnr=score))
     if best_initial is None or score>best_initial[0]:best_initial=(score,{k:v.detach().clone() for k,v in m.param.items()},velocity_factor)
 with torch.no_grad():
  for k,v in best_initial[1].items():m.param[k].copy_(v)
 m.set_velocity_factor(best_initial[2])
 for step in range(a.steps+1):
  if step%50==0 or step==a.steps:
   candidates=[]
   with torch.no_grad():
    for terminal in ['projected','integrated_tail']:
     for mode in modes:
      try:score=validate(mode,terminal);candidates.append(dict(mode=mode,terminal=terminal,foreground_psnr=score,completed=True))
      except (RuntimeError,ValueError) as error:candidates.append(dict(mode=mode,terminal=terminal,completed=False,error=str(error)))
   valid=[c for c in candidates if c['completed']];choice=max(valid,key=lambda c:c['foreground_psnr']) if valid else None
   if choice and (best is None or choice['foreground_psnr']>best['foreground_psnr']):
    best=dict(step=step,**choice);torch.save({k:v.detach().cpu().clone() for k,v in m.param.items()},out/'best.pt')
   history.append(dict(step=step,candidates=candidates,elapsed_seconds=time.time()-start));(out/'status.json').write_text(json.dumps(dict(completed=False,best=best,history=history,failures=failures),indent=2));print(step,best,flush=True)
  if step==a.steps:break
  f,cam,gt,mask=train[int(rng.integers(1,len(train)))];idx=max(0,int(np.searchsorted(adapter['times'],f['time']))-8);x=m.observed[idx].detach();v=m.observed_v[idx].detach();optimizer.zero_grad(set_to_none=True);old={k:v.detach().clone() for k,v in m.param.items()}
  try:
   x,v=m.integrate(x,v,float(m.times[idx]),f['time'],ad=True);pred=m.image(x,cam);loss=.4*(pred-gt).abs().mean()+.4*(pred-gt).abs()[:,mask].mean()+.2*(1-ssim(pred,gt));loads=m.param['loads'];loss=loss+1e-5*loads.square().mean()+1e-4*(loads[2:]-2*loads[1:-1]+loads[:-2]).square().mean()+1e-5*m.param['log_scale'].square();loss.backward();torch.nn.utils.clip_grad_norm_(m.param.parameters(),1.);optimizer.step();m.clamp()
   if not all(torch.isfinite(p).all() for p in m.param.values()):raise ValueError('Nonfinite physical parameter')
  except (RuntimeError,ValueError) as error:
   with torch.no_grad():
    for k,v in old.items():m.param[k].copy_(v)
   optimizer.state.clear();failures.append(dict(step=step,error=str(error)))
 if best is None:
  (out/'fit.json').write_text(json.dumps(dict(completed=False,reason='No stable validation rollout',history=history,failures=failures),indent=2));return
 seal=dict(completed=True,velocity_factor=best_initial[2],initial_candidates=initial_candidates,scene=a.scene,best=best,history=history,failures=failures,steps=a.steps,checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),appearance_sha256=hashlib.sha256((out/'appearance.pt').read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((out/'adapter.npz').read_bytes()).hexdigest(),test_images_read=False,future_force_labels_used=False,training='Fitting-prefix RGB multiple-shooting gradients with reconstructed segment initial states. Validation-only selection of checkpoint and four predeclared force-extrapolation rules.',scope='Actual tetrahedral forward integration; reconstructed volume, normalized mass/time, no calibrated SI forces. Fixed material-family decoder. No explicit inter-object contact in this adapter; unknown interactions represented by fitted Gaussian fields.')
 (out/'fit.json').write_text(json.dumps(seal,indent=2));print(best,flush=True)
if __name__=='__main__':main()
