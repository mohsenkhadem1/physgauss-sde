"""Compare sealed prefix models on genuinely later recorded frames."""
import sys,argparse,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import numpy as np,torch,lpips
 from PIL import Image
 from scripts.pidg_solver_forecast.common import BASE,DATA,camera,load,native
 from scripts.pidg_solver_forecast.model import SceneFEM
 from scripts.mechanics_gaussian_bridge.bridge import render
 from utils.loss_utils import ssim
 from scripts.pidg_solver_forecast.diagnostics import Interpenetration
 torch.set_num_threads(2);out=BASE/a.scene/'evaluation';out.mkdir(exist_ok=True);report=ROOT/'reports/pidg-solver-forecast-v01'/a.scene;report.mkdir(exist_ok=True);solver=BASE/a.scene/'solver';fit=json.loads((solver/'fit.json').read_text());assert not fit.get('test_images_read',False);adapter=np.load(solver/'adapter.npz');appearance=torch.load(solver/'appearance.pt',weights_only=False);m=SceneFEM(adapter,appearance);m.set_binding_tets();
 if fit['completed']:
  assert hashlib.sha256((solver/'best.pt').read_bytes()).hexdigest()==fit['checkpoint_sha256'];assert hashlib.sha256((solver/'appearance.pt').read_bytes()).hexdigest()==fit['appearance_sha256'];m.param.load_state_dict(torch.load(solver/'best.pt',map_location='cuda',weights_only=False))
 m.param.requires_grad_(False);contact_diagnostic=Interpenetration(adapter['elements'],adapter['component_labels'])
 checkpoints={}
 for arm in ['native-base','native-refined','hybrid']:
  run=BASE/a.scene/arm;meta=json.loads((run/'run.json').read_text());assert meta['status']=='training_complete';checkpoints[arm]={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'model').rglob('*') if p.is_file()}
 validation=json.loads((DATA/a.scene/'forecast_validation.json').read_text())['frames'];baseline_validation={}
 for arm in ['native-base','native-refined','hybrid']:
  g,d=load(a.scene,arm);scores=[]
  with torch.no_grad():
   for f in validation:
    cam,gt,mask=camera(a.scene,f);pred=native(g,d,cam,f['time'])['render'].clamp(0,1);scores.append(float(-10*(pred-gt).square()[:,mask].mean().clamp_min(1e-12).log10()))
  baseline_validation[arm]=float(np.mean(scores));del g,d;torch.cuda.empty_cache()
 native_choice=max(['native-base','native-refined'],key=lambda arm:baseline_validation[arm])
 cv_validation={}
 with torch.no_grad():
  cv_state=m.state(m.observed[-1]);
  for beta in [0.,.25,1.]:
   m.set_velocity_factor(beta);velocity=m.length*torch.einsum('pa,pai->pi',m.N,m.observed_v[-1][m.ids]);scores=[]
   for f in validation:
    cam,gt,mask=camera(a.scene,f);state={**cv_state,'centers':cv_state['centers']+(f['time']-m.end)*velocity};pred=render(cam,m.appearance,state,torch.zeros(3,device='cuda'))['render'].clamp(0,1);scores.append(float(-10*(pred-gt).square()[:,mask].mean().clamp_min(1e-12).log10()))
   cv_validation[beta]=float(np.mean(scores))
 cv_beta=max(cv_validation,key=cv_validation.get);m.set_velocity_factor(fit.get('velocity_factor',1.))

 seal=dict(cv_velocity_factor=cv_beta,cv_validation=cv_validation,solver=fit,baselines=checkpoints,baseline_validation=baseline_validation,native_selected=native_choice,test_targets_read=False);(out/'selection.json').write_text(json.dumps(seal,indent=2))
 # The following is the first final-test image read in this pipeline.
 frames=json.loads((DATA/a.scene/'forecast_test.json').read_text())['frames'];records=[(f,*camera(a.scene,f)) for f in frames];perceptual=lpips.LPIPS(net='alex').cuda().eval();rows=[];failures=[];x=m.observed[-1].clone();v=m.observed_v[-1].clone();t=m.end;states=[];times=[];physical_failed=not fit['completed']
 if physical_failed:failures.append(dict(method='solver',error=fit.get('reason','No stable validation model')))
 with torch.no_grad():
  initial=m.state(x);cv_v=m.rigid_v[-1]+cv_beta*(m.raw_v[-1]-m.rigid_v[-1]);vx=m.length*torch.einsum('pa,pai->pi',m.N,cv_v[m.ids]);initial={k:v.clone() for k,v in initial.items()}
  if fit['completed'] and fit['best'].get('terminal')=='integrated_tail':
   idx=max(0,len(m.times)-11);x,v=m.integrate(m.observed[idx].clone(),m.observed_v[idx].clone(),float(m.times[idx]),m.end)
 def score(pred,gt,mask,method,f):
  err=(pred-gt).square();union=mask|(pred.max(0).values>.03);fg=pred.max(0).values>.03;yy,xx=torch.meshgrid(torch.arange(gt.shape[1],device='cuda'),torch.arange(gt.shape[2],device='cuda'),indexing='ij');xy=torch.stack([xx,yy],-1).float();cgt=xy[mask].mean(0);cp=xy[fg].mean(0) if fg.any() else torch.full((2,),float('nan'),device='cuda')
  row=dict(method=method,frame=Path(f['file_path']).name,time=f['time'],psnr_db=float(-10*err.mean().clamp_min(1e-12).log10()),foreground_psnr_db=float(-10*err[:,mask].mean().clamp_min(1e-12).log10()),union_psnr_db=float(-10*err[:,union].mean().clamp_min(1e-12).log10()),ssim=float(ssim(pred,gt)),lpips_alex=float(perceptual(pred[None]*2-1,gt[None]*2-1)),silhouette_centroid_error_px=float((cp-cgt).norm()),silhouette_iou=float((fg&mask).sum()/(fg|mask).sum().clamp_min(1)))
  folder=out/method;folder.mkdir(exist_ok=True);Image.fromarray((pred.cpu().numpy().transpose(1,2,0)*255).round().astype('uint8')).save(folder/(row['frame']+'.png'));rows.append(row);return row
 with torch.no_grad():
  for f,cam,gt,mask in records:
   folder=out/'reference';folder.mkdir(exist_ok=True);name=Path(f['file_path']).name;Image.fromarray((gt.cpu().numpy().transpose(1,2,0)*255).round().astype('uint8')).save(folder/(name+'.png'));Image.fromarray(mask.cpu().numpy().astype('uint8')*255).save(folder/(name+'-mask.png'))
   state={**initial,'centers':initial['centers']+(f['time']-m.end)*vx};pred=render(cam,m.appearance,state,torch.zeros(3,device='cuda'))['render'].clamp(0,1);score(pred,gt,mask,'constant-velocity',f)
   if not physical_failed:
    try:
     x,v=m.integrate(x,v,t,f['time'],fit['best']['mode']);t=f['time'];pred=m.image(x,cam);r=score(pred,gt,mask,'solver',f);J=torch.linalg.det(m.deformation(x));r.update(minimum_J=float(J.min()),inversion_fraction=float((J<=0).float().mean()),mean_abs_volume_change=float((J-1).abs().mean()),net_internal_force=float(m.bulk(x,scale=m.param['log_scale'].exp()/12000)[0].sum(0).norm()));r.update(contact_diagnostic(x));states.append(x.cpu().numpy());times.append(t)
    except (ValueError,RuntimeError) as error:physical_failed=True;failures.append(dict(method='solver',frame=name,error=str(error)))
 for arm in ['native-refined','hybrid']:
  g,d=load(a.scene,native_choice if arm=='native-refined' else arm)
  with torch.no_grad():
   for f,cam,gt,mask in records:score(native(g,d,cam,f['time'])['render'].clamp(0,1),gt,mask,arm,f)
  del g,d;torch.cuda.empty_cache()
 if states:np.savez_compressed(out/'solver-states.npz',x=states,times=times)
 summary={}
 for method in ['native-refined','hybrid','solver','constant-velocity']:
  rr=[r for r in rows if r['method']==method];summary[method]=dict(completed_frames=len(rr),expected_frames=len(frames),complete=len(rr)==len(frames),metrics={k:float(np.mean([r[k] for r in rr])) for k in ['psnr_db','foreground_psnr_db','union_psnr_db','ssim','lpips_alex','silhouette_centroid_error_px','silhouette_iou']} if rr else {})
 result=dict(scene=a.scene,cv_velocity_factor=cv_beta,native_selected=native_choice,baseline_validation=baseline_validation,summary=summary,rows=rows,failures=failures,scope='Chronological forecast, first 60% images only for training; 60-70% validation selection, released test frames after 70%. Previously exposed dataset. Shared prefix appearance initialization; solver appearance frozen, native/hybrid 5k refinements. Geometry/mass/time reconstructed/normalized. Centroid and optical flow are image-space proxies, not ground-truth 3D mechanics. No measured-force/OOD intervention claims.')
 (out/'results.json').write_text(json.dumps(result,indent=2));(report/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
