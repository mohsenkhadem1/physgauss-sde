import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_appearance_transfer.model import Transfer,BASE
from scripts.pidg_mechanics.appearance import camera_frame
m=Transfer();checkpoint=torch.load(BASE/'joint-best.pt',weights_only=False);m.restore(checkpoint['parameters']);d=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_test.json').read_text());rows=[]
from scripts.mechanics_gaussian_bridge.bridge import render
for f in d['frames']:
 camera,gt,mask,frame=camera_frame(f,d);state={k:v.detach() for k,v in m.state(frame,True).items()}
 for mode in ['material','world']:
  s=dict(state)
  if mode=='world':s['rotation']=torch.eye(3,device='cuda').expand(len(m.features),3,3)
  with torch.no_grad():pred=render(camera,m,s,torch.zeros(3,device='cuda'))['render'].clamp(0,1);e=(pred-gt).square();rows.append({'frame':frame,'mode':mode,'psnr':float(-10*e.mean().log10()),'foreground_psnr':float(-10*e[:,mask].mean().log10())})
r={mode:{k:float(np.mean([x[k] for x in rows if x['mode']==mode])) for k in ['psnr','foreground_psnr']} for mode in ['material','world']};(BASE/'radiance-validation.json').write_text(json.dumps({'checkpoint_step':checkpoint['step'],'summary':r,'rows':rows,'test_images_read':False},indent=2));print(r)
