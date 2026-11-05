import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(int(os.environ.get('PIDG_GPU','0')));sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_appearance_transfer.model import Transfer,BASE
from scripts.pidg_mechanics.appearance import camera_frame

def main():
 torch.set_num_threads(2);torch.manual_seed(44);m=Transfer();base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit';tr=json.loads((base/'transforms_train.json').read_text());va=json.loads((base/'transforms_test.json').read_text());train=[camera_frame(f,tr) for f in tr['frames']];val=[camera_frame(f,va) for f in va['frames']];live=os.environ.get('TRANSFER_LIVE','0')=='1';label='joint' if live else 'appearance';steps=int(os.environ.get('TRANSFER_STEPS','2000'));every=int(os.environ.get('TRANSFER_EVERY','200'));history=[];start=time.time();best=-np.inf
 if live:m.restore(torch.load(BASE/'appearance-best.pt',weights_only=False)['parameters'])
 if os.environ.get('TRANSFER_RESUME')=='1':m.restore(torch.load(BASE/'joint-best.pt',weights_only=False)['parameters'])
 cached_live=os.environ.get('TRANSFER_CACHE_LIVE')=='1'
 if cached_live:
  m.restore(torch.load(BASE/'joint-best.pt',weights_only=False)['parameters']);label='refined';live=False
  for f in sorted({x[3] for x in train+val}):m.cache[f]={k:v.detach() for k,v in m.raw(f,m.solve(f)).items()}
  if os.environ.get('TRANSFER_RESET_APPEARANCE')=='1':
   with torch.no_grad():m.features.copy_(m.source['features'].cuda());m.opacity_raw.copy_(torch.logit(m.source['opacity'].cuda().clamp(1e-6,1-1e-6)))
 frozen_controls={k:getattr(m,k).detach().clone() for k in ['controls','patch_controls']}
 m.controls.requires_grad_(live);m.patch_controls.requires_grad_(live)
 opt=torch.optim.Adam([{'params':[m.features],'lr':.003 if live else .008},{'params':[m.opacity_raw,m.offset,m.scale],'lr':.002},{'params':[m.controls,m.patch_controls] if live else [],'lr':.012}]);original=m.features.detach().clone()
 for step in range(steps+1):
  if step:
   cam,gt,mask,f=train[(step*37)%len(train)];pred=m.image(cam,f,live);loss=(pred-gt).abs().mean()+.3*(pred[:,mask]-gt[:,mask]).square().mean()+1e-6*(m.features-original).square().mean()+.0001*m.offset.square().mean()+.0001*m.controls.square().mean()+.0001*m.patch_controls.square().mean()+.0003*(m.patch_controls[1:]-m.patch_controls[:-1]).square().mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.);opt.step()
  if step%every==0 or step==steps:
   values=[]
   for cam,gt,mask,f in val:
    pred=m.image(cam,f,live).detach().clamp(0,1);e=(pred-gt).square();values.append([float(-10*e.mean().log10()),float(-10*e[:,mask].mean().log10())])
   means=np.mean(values,0);row=dict(step=step,validation_psnr=float(means[0]),validation_foreground_psnr=float(means[1]),seconds=time.time()-start);history.append(row)
   if means[1]>best:best=means[1];torch.save({'parameters':m.parameters_only(),'step':step,'live':live or cached_live},BASE/f'{label}-best.pt')
   (BASE/f'{label}-progress.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 if not live:assert all(torch.equal(getattr(m,k),v) for k,v in frozen_controls.items()),'Frozen boundary controls changed'
 (BASE/f'{label}-complete.json').write_text(json.dumps({'completed':True,'steps':steps,'live_mechanics':live or cached_live,'cached_mechanics_during_appearance':cached_live,'validation_foreground_psnr':best,'test_images_read':False},indent=2))
if __name__=='__main__':main()
