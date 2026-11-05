"""Sealed comparison after validation selection; no fitting or target-view synthesis."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(int(os.environ.get('PIDG_GPU','2')));sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from PIL import Image
from scripts.pidg_mechanics.appearance import Appearance,camera_frame
from scripts.pidg_mechanics.joint import Joint
from utils.loss_utils import ssim
import lpips
# joint import configures the same explicitly selected physical device.
root=ROOT/os.environ.get('PIDG_MECHANICS_ROOT','runs/pidg-mechanics-v03');seal=json.loads((root/'final-selection.json').read_text());assert seal['completed']
a=Appearance();chosen=torch.load(root/seal['checkpoint'],weights_only=False)
with torch.no_grad():
 for k in ['features','radius_raw','scale_raw','opacity_raw']:getattr(a,k).copy_(chosen['appearance'][k] if seal['joint'] else chosen[k])
j=Joint(a)
if seal['joint']:
 with torch.no_grad():j.controls.copy_(chosen['controls']);j.material.copy_(chosen['material'])
perceptual=lpips.LPIPS(net='alex').cuda().eval();doc=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction/evaluation/transforms_test.json').read_text());out=root/'evaluation';out.mkdir(exist_ok=True);rows=[];states={};mechanics_diagnostics=[]
from scripts.mechanics_gaussian_bridge.bridge import render
for f in doc['frames']:
 index=int(Path(f['file_path']).name)-1
 # Differentiable constitutive residual is needed inside the equilibrium solve;
 # detach the completed physical state before image scoring.
 state=j.state(index) if seal['joint'] else a.state(index);state={k:v.detach() for k,v in state.items()};states[index]=state
 if seal['joint']:
  for b,m in enumerate(a.models):
   F=torch.einsum('eai,eaJ->eiJ',m.last_solved_state[m.elements],m.grad);J=torch.linalg.det(F);mechanics_diagnostics.append({'frame':index+1,'body':b,'minimum_J':float(J.min()),'inverted_tetrahedra':int((J<=0).sum()),'tetrahedra':len(J),'volume_ratio':float((m.vol*J).sum()/m.vol.sum()),'solver':m.last_diagnostic})
 for width in [320,640,1600]:
  cam,gt,mask,_=camera_frame(f,doc,width)
  with torch.no_grad():
   pred=render(cam,a,state,torch.zeros(3,device='cuda'))['render'].clamp(0,1);error=(pred-gt).square();row={'frame':Path(f['file_path']).name,'width':width,'psnr_db':float(-10*error.mean().clamp_min(1e-12).log10()),'foreground_psnr_db':float(-10*error[:,mask].mean().clamp_min(1e-12).log10()),'foreground_rmse':float(error[:,mask].mean().sqrt()),'ssim':float(ssim(pred,gt)),'lpips_alex':float(perceptual(pred[None]*2-1,gt[None]*2-1).mean())}
  rows.append(row)
  if width==640:
   Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/f"{index+1:04d}.png")
 means={str(w):{k:float(np.mean([r[k] for r in rows if r['width']==w])) for k in ['psnr_db','foreground_psnr_db','foreground_rmse','ssim','lpips_alex']} for w in [320,640,1600]}
 (out/'metrics-progress.json').write_text(json.dumps({'rows':rows,'means':means},indent=2));print(index+1,flush=True)
novel=[]
for ii in [0,len(doc['frames'])//2,len(doc['frames'])-1]:
 f=doc['frames'][ii];index=int(Path(f['file_path']).name)-1
 for angle in [-90,-60,-30,0,30,60,90]:
  camera,_,_,_=camera_frame(f,doc,640,angle)
  with torch.no_grad():pred=render(camera,a,states[index],torch.zeros(3,device='cuda'))['render'].clamp(0,1)
  name=f"{index+1:04d}-yaw{angle:+04d}.png";Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/name);novel.append({'image':name,'frame':index+1,'time':f['time'],'yaw_deg':angle,'metric_available':angle==0})
(out/'results.json').write_text(json.dumps({'completed':True,'selection':seal,'means':means,'rows':rows,'novel_views':novel,'mechanics_diagnostics':mechanics_diagnostics,'novel_view_limit':'No reference images at nonzero angles; no new-view PSNR or accuracy winner claimed.','method':'Paper NeuralFEM equilibrium adapter with inferred Gaussian attachment targets, fixed learned material decoder, mechanics-bound Gaussian appearance','scope':'Monocular six-ball geometry prior and position-conditioned reconstruction. Not full coupled transient solver, recovered ground-truth physics, or changed-action forecast.'},indent=2));print(json.dumps(means),flush=True)
