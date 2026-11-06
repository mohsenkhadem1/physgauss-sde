"""Preserve dense detail with prefix-only static appearance refinement.
Motion remains mechanically bound; no temporal Gaussian residual is introduced.
"""
import sys,argparse,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--steps',type=int,default=2000);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from types import SimpleNamespace
 from scripts.pidg_solver_forecast.common import BASE,DATA,camera,load,native
 from scripts.pidg_solver_forecast.model import SceneFEM
 from scripts.mechanics_gaussian_bridge.bridge import render
 from utils.loss_utils import ssim
 torch.set_num_threads(2);rng=np.random.default_rng(17);out=BASE/a.scene/'solver';adapter=np.load(out/'adapter.npz');appearance=torch.load(out/'appearance.pt',weights_only=False);m=SceneFEM(adapter,appearance);m.set_binding_tets();m.param.requires_grad_(False);original={k:v.clone() if torch.is_tensor(v) else v for k,v in appearance.items()};records=[]
 for split,file in [('fit','fit'),('validation','appearance_validation')]:
  for f in json.loads((DATA/a.scene/(file+'.json')).read_text())['frames']:
   j=max(0,min(len(adapter['times'])-2,np.searchsorted(adapter['times'],f['time'])-1));alpha=np.clip((f['time']-adapter['times'][j])/(adapter['times'][j+1]-adapter['times'][j]),0,1);x=m.observed[j]*(1-alpha)+m.observed[j+1]*alpha
   with torch.no_grad():state={k:v.detach() for k,v in m.state(x).items()}
   cam,gt,mask=camera(a.scene,f);records.append(dict(split=split,state=state,cam=cam,gt=gt,mask=mask))
 features=torch.nn.Parameter(m.appearance.get_features.clone());logit=torch.nn.Parameter(torch.logit(m.appearance.get_opacity.clamp(.001,.999)));logscale=torch.nn.Parameter(torch.zeros(len(m.centers),device='cuda'));opt=torch.optim.Adam([{'params':[features],'lr':.002},{'params':[logit],'lr':.01},{'params':[logscale],'lr':.001}]);train=[r for r in records if r['split']=='fit'];val=[r for r in records if r['split']=='validation'];best=None;history=[];start=time.time()
 g,d=load(a.scene);native_prefix=[]
 with torch.no_grad():
  for r in val:
   pred=native(g,d,r['cam'],float(r['cam'].fid))['render'].clamp(0,1);native_prefix.append(float(-10*(pred-r['gt']).square()[:,r['mask']].mean().clamp_min(1e-12).log10()))
 del g,d;torch.cuda.empty_cache()

 radiance_frame='material'
 def image(r):
  ap=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,get_features=features,get_opacity=logit.sigmoid());state={**r['state'],'covariance':r['state']['covariance']*logscale.exp().square()[:,None,None]}
  if radiance_frame=='world':state['rotation']=torch.eye(3,device='cuda').expand_as(state['rotation'])
  return render(r['cam'],ap,state,torch.zeros(3,device='cuda'))['render'].clamp(0,1)
 angular_candidates={}
 with torch.no_grad():
  for radiance_frame in ['material','world']:
   angular_candidates[radiance_frame]=float(np.mean([float(-10*(image(r)-r['gt']).square()[:,r['mask']].mean().clamp_min(1e-12).log10()) for r in val]))
 radiance_frame=max(angular_candidates,key=angular_candidates.get)
 for step in range(a.steps+1):
  if step%250==0 or step==a.steps:
   with torch.no_grad():score=float(np.mean([float(-10*(image(r)-r['gt']).square()[:,r['mask']].mean().clamp_min(1e-12).log10()) for r in val]))
   history.append(dict(step=step,validation_foreground_psnr=score,seconds=time.time()-start))
   if best is None or score>best['score']:
    best=dict(step=step,score=score);saved={**original,'radiance_frame':radiance_frame,'features':features.detach().cpu().clone(),'opacity':logit.sigmoid().detach().cpu().clone(),'covariance':original['covariance']*logscale.detach().cpu().exp().square()[:,None,None]};torch.save(saved,out/'appearance-refined.pt')
   print(history[-1],flush=True)
  if step==a.steps:break
  r=train[int(rng.integers(len(train)))];opt.zero_grad(set_to_none=True);pred=image(r);loss=.4*(pred-r['gt']).abs().mean()+.4*(pred-r['gt']).abs()[:,r['mask']].mean()+.2*(1-ssim(pred,r['gt']))+.001*logscale.square().mean();loss.backward();opt.step()
  with torch.no_grad():logscale.clamp_(-.7,.7);logit.clamp_(-12,12)
 torch.save(original,out/'appearance-initial.pt');(out/'appearance.pt').write_bytes((out/'appearance-refined.pt').read_bytes());report=dict(completed=True,radiance_frame=radiance_frame,angular_candidates=angular_candidates,native_prefix_validation_foreground_psnr=float(np.mean(native_prefix)),best=best,history=history,steps=a.steps,appearance_sha256=hashlib.sha256((out/'appearance.pt').read_bytes()).hexdigest(),future_images_read=False,scope='Static SH/opacity/covariance-size refinement from prefix RGB, with frozen reconstructed nodal motion and fixed Gaussian binding. No temporal residual. Prefix-only validation selects initialization or refined appearance; fixed thereafter.')
 (out/'appearance-fit.json').write_text(json.dumps(report,indent=2));print(best)
if __name__=='__main__':main()
