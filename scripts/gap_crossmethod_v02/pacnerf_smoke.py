"""Native PAC-NeRF execution with explicit bounded development budgets.
Imports published classes/training functions unchanged; no shared FEM substitute.
"""
from pathlib import Path
import os,json,time,hashlib,sys,argparse
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-crossmethod-v02';WORK=BASE/'pacnerf-work';REPORT=ROOT/'reports/gap-crossmethod-v02';os.chdir(WORK)
import torch,numpy as np,mmcv
import train as native
ap=argparse.ArgumentParser();ap.add_argument('--static-steps',type=int,default=101);a=ap.parse_args();torch.set_num_threads(4)
report=REPORT/'pacnerf-smoke.json';status=dict(scope='Native PAC-NeRF author code with reduced development iteration budget; not final accuracy benchmark',static_steps=a.static_steps,fit_cameras=list(range(1,11)),heldout_camera=0,fit_frames=[0],started=time.time(),native_commit='b613048557d0648e885697ececbef80297defac0',phase='preprocess')
def save():report.write_text(json.dumps(status,indent=2))
save()
try:
 cfg=mmcv.Config.fromfile(str(BASE/'vendor/pac-nerf/configs/torus.py'))['cfg'];cfg['base_dir']=str(WORK/f'checkpoint/torus-static{a.static_steps}');cfg['data_dir']=str(WORK/'data/torus');cfg['taichi_cuda_memory']=.30;cfg['nerf_bs']=16384;cfg['N_static']=a.static_steps
 data=native.load_data(cfg['data_dir'],cfg['n_cameras'],cfg['H'],cfg['W']);status.update(phase='static_fit',shapes=[list(x.shape) for x in data],cfg=dict(cfg));save()
 model,opt,start=native.create_model_and_optimizer(cfg,'static');native.cfg=cfg
 fit_data=tuple(x[1:] for x in data)  # camera0 never enters optimization
 native.train_static(cfg,model,opt,start,a.static_steps,*fit_data)
 status.update(phase='render_static');save()
 rgbs,depth,bg=model.render_sequence(1,H=cfg['H'],W=cfg['W'],c2w=None,K=None,rays_o=data[0][0],rays_d=data[1][0],viewdirs=data[2][0],savedir=cfg['base_dir']+'/image',dump_images=True)
 truth=data[3][0,0].reshape(cfg['H'],cfg['W'],3).numpy();pred=np.array(rgbs)[0];mse=float(np.mean((pred-truth)**2));status.update(phase='complete',static_frame0_psnr=-10*np.log10(mse),seconds=time.time()-status['started'],render_shape=list(pred.shape));save()
except Exception as e:
 import traceback
 status.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
