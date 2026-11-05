"""Evaluate validation-selected transferred mechanics, without further optimization."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(int(os.environ.get('PIDG_GPU','0')));sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from PIL import Image
from scripts.pidg_appearance_transfer.model import Transfer,BASE
from scripts.pidg_mechanics.appearance import camera_frame
from scripts.mechanics_gaussian_bridge.bridge import render
from utils.loss_utils import ssim
import lpips
seal=json.loads((ROOT/'reports/pidg-appearance-transfer-v01/selection.json').read_text());assert str(BASE.relative_to(ROOT))==seal['run'];path=BASE/seal['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['checkpoint_sha256'];saved=torch.load(path,weights_only=False);m=Transfer();m.restore(saved['parameters']);metric=lpips.LPIPS(net='alex').cuda().eval();torch.set_num_threads(2);out=BASE/'evaluation';out.mkdir(exist_ok=True);d=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction/evaluation/transforms_test.json').read_text());rows=[];cache={};diagnostics=[]
for f in d['frames']:
 index=int(Path(f['file_path']).name)-1;state={k:v.detach() for k,v in m.state(index,saved['live']).items()};cache[index]=state
 for b,body in enumerate(m.models):
  if hasattr(body,'last_diagnostic'):diagnostics.append({'frame':index,'body':b,**body.last_diagnostic})
 for width in [320,640,1600]:
  camera,gt,mask,_=camera_frame(f,d,width)
  with torch.no_grad():
   pred=render(camera,m,state,torch.zeros(3,device='cuda'))['render'].clamp(0,1);err=(pred-gt).square();row={'frame':Path(f['file_path']).name,'width':width,'psnr_db':float(-10*err.mean().clamp_min(1e-12).log10()),'foreground_psnr_db':float(-10*err[:,mask].mean().clamp_min(1e-12).log10()),'ssim':float(ssim(pred,gt)),'lpips_alex':float(metric(pred[None]*2-1,gt[None]*2-1).mean())};rows.append(row)
  if width==640:
   Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/f'{index+1:04d}.png');np.savez_compressed(out/f'{index+1:04d}.npz',prediction=pred.permute(1,2,0).cpu().numpy(),truth=gt.permute(1,2,0).cpu().numpy(),mask=mask.cpu().numpy())
 (out/'progress.json').write_text(json.dumps(rows));print(index+1,flush=True)
novel=[]
for k in [0,len(d['frames'])//2,len(d['frames'])-1]:
 f=d['frames'][k];index=int(Path(f['file_path']).name)-1
 for angle in [-90,-60,-30,0,30,60,90]:
  camera,_,_,_=camera_frame(f,d,640,angle)
  with torch.no_grad():pred=render(camera,m,cache[index],torch.zeros(3,device='cuda'))['render'].clamp(0,1)
  name=f'{index+1:04d}-yaw{angle:+04d}.png';Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/name);novel.append({'frame':index+1,'time':f['time'],'yaw_deg':angle,'image':name,'ground_truth_available':angle==0})
means={str(width):{key:float(np.mean([r[key] for r in rows if r['width']==width])) for key in ['psnr_db','foreground_psnr_db','ssim','lpips_alex']} for width in [320,640,1600]};r={'completed':True,'selection':seal,'means':means,'rows':rows,'novel_views':novel,'mechanics_diagnostics':diagnostics,'limits':'Reconstruction-assisted equilibrium adapter; no coupled collision/new-action or true material inference claim. Unrecorded cameras lack reference RGB. Released test split was reused in prior development.'};(out/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps(means),flush=True)
