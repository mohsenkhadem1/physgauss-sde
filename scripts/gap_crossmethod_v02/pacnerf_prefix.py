"""Native PAC-NeRF prefix fit and withheld-view/future evaluation.
Published training/solver/rendering functions are imported without modifications.
"""
from pathlib import Path
import os,json,time,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-crossmethod-v02';WORK=BASE/'pacnerf-work';REPORT=ROOT/'reports/gap-crossmethod-v02';os.chdir(WORK)
import torch,numpy as np,mmcv,imageio.v2 as imageio
import train as native
P=REPORT/'pacnerf-prefix-protocol.json';protocol=json.loads(P.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f,h in protocol['source_sha256'].items():assert sha(ROOT/f)==h,f
assert sha(BASE/'pacnerf-inputs/data-manifest.json')==protocol['data_manifest_sha256']
torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123)
statepath=REPORT/'pacnerf-prefix-status.json';status=dict(started=time.time(),phase='loading',protocol_sha256=sha(P))
def save():statepath.write_text(json.dumps(status,indent=2))
save()
try:
 cfg=mmcv.Config.fromfile(str(BASE/'vendor/pac-nerf/configs/torus.py'))['cfg'];cfg.update(protocol['config_overrides']);native.cfg=cfg
 data=native.load_data(cfg['data_dir'],11,H=800,W=800)
 fit_data=(data[0][1:],data[1][1:],data[2][1:],data[3][1:,:8],data[4][1:])
 for stage in ['static','velocity','dynamic']:
  status.update(phase=stage,seconds=time.time()-status['started']);save()
  if stage=='static':
   model,opt,start=native.create_model_and_optimizer(cfg,stage,pnerf=None);native.train_static(cfg,model,opt,start,cfg['N_static'],*fit_data)
  elif stage=='velocity':
   model,opt,start=native.create_model_and_optimizer(cfg,stage,pnerf=model);native.train_velocity(cfg,model,start,*fit_data)
  else:
   model,opt,start=native.create_model_and_optimizer(cfg,stage,pnerf=model);native.train_dynamic(cfg,model,opt,start,*fit_data)
 status.update(phase='withheld_render');save()
 # All fitting has finished before camera0 labels are used for metrics.
 rgbs,_,_=model.render_sequence(14,H=800,W=800,c2w=None,K=None,rays_o=data[0][0],rays_d=data[1][0],viewdirs=data[2][0],savedir=cfg['base_dir']+'/heldout-camera0',dump_images=True)
 pred=np.asarray(rgbs);truth=data[3][0].reshape(14,800,800,3).numpy();out=BASE/'pacnerf-predictions';out.mkdir(exist_ok=True);np.savez_compressed(out/'camera0.npz',prediction=pred,reference=truth)
 rows=[]
 for i,(x,y) in enumerate(zip(pred,truth)):
  err=(x-y)**2;mask=np.min(y,axis=-1)<.95;mse=float(err.mean());fg=float(err[mask].mean()) if mask.any() else None
  rows.append(dict(frame=i,split='held_view' if i<8 else 'future_and_held_view',psnr_db=float(-10*np.log10(mse)),foreground_psnr_db=None if fg is None else float(-10*np.log10(fg)),foreground_pixels=int(mask.sum())))
 metrics=dict(protocol_sha256=sha(P),rows=rows,learned_parameters=dict(E=float(10**model.global_E.item()),nu=float(native.constraint(model.global_nu,model.nu_bound).item()) if hasattr(native,'constraint') else None,initial_velocity=model.global_v.detach().cpu().tolist()),seconds=time.time()-status['started'],scope='Native method with explicit prefix/view split; no improvement claim without matched comparison')
 (REPORT/'pacnerf-prefix-results.json').write_text(json.dumps(metrics,indent=2));imageio.mimwrite(out/'camera0.mp4',(np.clip(pred,0,1)*255).astype(np.uint8),fps=24,quality=8)
 status.update(phase='complete',seconds=time.time()-status['started']);save()
except Exception as e:
 import traceback
 status.update(phase='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-status['started']);save();raise
