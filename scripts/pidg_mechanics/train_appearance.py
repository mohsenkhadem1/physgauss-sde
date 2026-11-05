import os
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(int(os.environ.get('PIDG_GPU','0')));sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_mechanics.appearance import Appearance,camera_frame
base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit';out=ROOT/os.environ.get('PIDG_MECHANICS_ROOT','runs/pidg-mechanics-v01');torch.manual_seed(0);torch.set_num_threads(2)
train=json.loads((base/'transforms_train.json').read_text());val=json.loads((base/'transforms_test.json').read_text());training=[camera_frame(f,train) for f in train['frames']];validation=[camera_frame(f,val) for f in val['frames']];model=Appearance();opt=torch.optim.Adam([{'params':[model.features],'lr':.015},{'params':[model.radius_raw,model.scale_raw,model.opacity_raw],'lr':.003}]);best=-float('inf');history=[];start=time.time()
for step in range(2001):
 if step:
  camera,gt,mask,frame=training[torch.randint(len(training),()).item()];pred=model.image(camera,frame)
  loss=(pred-gt).abs().mean()+.3*(pred[:,mask]-gt[:,mask]).square().mean()+.0001*model.features[:,1:].square().mean()+.0001*model.radius_raw.square().mean()
  opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
 if step%200==0:
  with torch.no_grad():
   metrics=[]
   for camera,gt,mask,frame in validation:
    pred=model.image(camera,frame).clamp(0,1);err=(pred-gt).square();metrics.append([float(-10*err.mean().clamp_min(1e-12).log10()),float(-10*err[:,mask].mean().clamp_min(1e-12).log10())])
  mean=np.mean(metrics,0);row={'step':step,'validation_psnr':float(mean[0]),'validation_foreground_psnr':float(mean[1]),'seconds':time.time()-start};history.append(row)
  if mean[1]>best:
   best=mean[1];torch.save({'features':model.features.detach(),'radius_raw':model.radius_raw.detach(),'scale_raw':model.scale_raw.detach(),'opacity_raw':model.opacity_raw.detach(),'step':step},out/'appearance-best.pt')
  (out/'appearance-progress.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
(out/'appearance-selection.json').write_text(json.dumps({'completed':True,'criterion':'Highest validation foreground PSNR at fixed 200-step checkpoints','test_images_read':False,'best_foreground_psnr':best,'training_steps':2000,'learned':'SH appearance, opacity >=.65, bounded radial offsets and covariance scales; mechanics frozen after training-position fit.'},indent=2))
