"""Position-only equilibrium fit with learnable physical attachment patch width.
Same inherited Neo-Hookean mesh/material fields and strict J cutoff. Width is
an effective attachment hypothesis, not an identified physical clamp label.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.external_benchmark.fem import ExternalFEM

class AttachmentFEM(ExternalFEM):
 def __init__(self,data,device='cuda:1',**kwargs):
  if 'evaluation' in data:raise ValueError('Evaluator forbidden')
  super().__init__(data,device=device,**kwargs)
  self.original_weights=self.weights.detach().clone()
  self.log_patch=torch.log(self.original_weights.clamp_min(1e-300))
  self.width_index=self.i_field+1+int(self.contact)
  self.width_count=len(self.weights)
 def evaluate(self,x,theta,frame,tangent=False):
  if len(theta)!=self.width_index+self.width_count:raise ValueError('Attachment parameter count')
  # Original Gaussian has sigma15mm. Precision in [.25,4] means sigma30..7.5mm.
  self.weights=torch.softmax(self.log_patch*torch.exp(theta[self.width_index:])[:,None],dim=1)
  return super().evaluate(x,theta,frame,tangent)

def fit(data,device='cuda:1',iterations=100,frames=36,budget_seconds=1800,check_callback=None,checkpoint_callback=None):
 if 'evaluation' in data or len(data['observed'])!=data['train_end']:raise ValueError('Strict input-only prefix required')
 model=AttachmentFEM(data,device=device,field=True,contact=True,nonconvex=True,materials=6)
 end=data['train_end'];selected=np.unique(np.linspace(0,end-1,frames,dtype=int));started=time.monotonic()
 def check():
  if check_callback is not None:check_callback()
  if time.monotonic()-started>budget_seconds:raise TimeoutError('Declared attachment fit budget')
 A=model.t(data['observation_matrix']);y=model.t(data['observed']/model.scale);initial=model.t(data['observed_initial']/model.scale)
 mask=torch.as_tensor(data['observed_mask'],device=device);n=model.width_count
 theta0=np.array([-4]+[2]*n+[5]+[0]*6+[0,1]+[0]*n,float)
 bounds=[(-9.,1.)]+[(-3.,8.)]*n+[(0.,9.)]+[(-2.,2.)]*6+[(-3.,3.),(-5.,7.)]+[(-np.log(4),np.log(4))]*n
 best={'loss':float('inf')};history=[];failures=[]
 def objective(values):
  check();th=model.t(values).requires_grad_();x=None;losses=[]
  try:
   for frame in selected:
    check();x=model.predict(th,int(frame),None if x is None else x.detach());prediction=initial+A@(x-model.rest);valid=mask[frame]
    if valid.any():losses.append(((prediction-y[frame])[valid]**2).mean())
   fitting=torch.stack(losses).mean();regularizer=1e-6*(th[model.i_material:model.i_field+1]**2).sum()+1e-6*(th[model.width_index:]**2).sum()
   loss=fitting+regularizer;loss.backward();value=float(loss.detach());gradient=th.grad.detach().cpu().numpy()
   if not np.isfinite(value) or not np.isfinite(gradient).all():raise RuntimeError('Nonfinite objective')
   record=dict(call=len(history),loss=value,fit_rmse_mm=float(torch.sqrt(fitting.detach()))*model.scale*1000,theta=values.tolist(),elapsed_seconds=time.monotonic()-started)
   history.append(record)
   if value<best['loss']:
    best.update(loss=value,theta=values.copy(),record=record)
    if checkpoint_callback is not None:checkpoint_callback(values.copy(),record,selected.copy())
   print('attachment',end,record['call'],record['fit_rmse_mm'],flush=True)
   return value,gradient
  except (RuntimeError,ValueError) as e:
   failures.append(dict(theta=values.tolist(),reason=str(e)));delta=values-best.get('theta',theta0);return 1e3+float(delta@delta),2*delta
 try:
  result=minimize(objective,theta0,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=iterations,maxls=12,ftol=1e-10,gtol=1e-7,maxfun=iterations*6))
  optimizer=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
 except TimeoutError:optimizer=dict(success=False,message='Budget exhausted; best valid prefix checkpoint retained')
 if 'theta' not in best:raise RuntimeError('No valid attachment fitting iterate')
 return best['theta'],dict(fit_end=end,selected_frames=selected.tolist(),best=best['record'],history=history,failed_trials=failures,optimizer=optimizer,elapsed_seconds=time.monotonic()-started,width_mm=(15*np.exp(-.5*best['theta'][model.width_index:])).tolist(),material_count=6,volume_cutoff=.03,geometry='Unchanged inherited nonconvex mesh',scope='Effective quasistatic position-conditioned attachment/material ratios, not force/material truth')

def predict(data,theta,device='cuda:1',budget_seconds=1200,check_callback=None):
 if 'evaluation' in data:raise ValueError('Evaluator forbidden')
 model=AttachmentFEM(data,device=device,field=True,contact=True,nonconvex=True,materials=6);th=model.t(theta);x=None;pred=[];rows=[];start=time.monotonic()
 for frame in range(len(data['times'])):
  if check_callback is not None:check_callback()
  if time.monotonic()-start>budget_seconds:
   for pending in range(frame,len(data['times'])):
    pred.append(np.full_like(data['nodes'],np.nan));rows.append(dict(failed=True,reason='Declared attachment forecast budget exhausted'))
   break
  try:
   x,H,diagnostic=model.solve(th,frame,x);p=x.cpu().numpy()*model.scale+model.origin
  except (RuntimeError,ValueError) as e:p=np.full_like(data['nodes'],np.nan);diagnostic=dict(failed=True,reason=str(e))
  pred.append(p);rows.append(diagnostic)
 return np.asarray(pred),dict(diagnostics=rows,failed_frames=[i for i,r in enumerate(rows) if r.get('failed')],elapsed_seconds=time.monotonic()-start)
