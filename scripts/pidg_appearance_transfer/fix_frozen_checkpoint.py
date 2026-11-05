"""Repair a serialization defect, preserving exactly the validation-rendered geometry."""
import os,sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(int(os.environ.get('PIDG_GPU','0')));sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_appearance_transfer.model import Transfer,BASE
from scripts.pidg_mechanics.appearance import camera_frame
torch.set_num_threads(2)
p=BASE/'refined-best.pt';joint=torch.load(BASE/'joint-best.pt',weights_only=False);d=torch.load(p,weights_only=False);backup=BASE/'refined-before-freeze-fix.pt'
if not backup.exists():shutil.copyfile(p,backup)
for key in ['controls','patch_controls']:
 if key in joint['parameters']:d['parameters'][key]=joint['parameters'][key].clone()
 elif key in d['parameters']:d['parameters'][key].zero_()
m=Transfer();m.restore(d['parameters']);va=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_test.json').read_text());values=[]
for f in va['frames']:
 cam,gt,mask,index=camera_frame(f,va);pred=m.image(cam,index,True).detach().clamp(0,1);e=(pred-gt).square();values.append([float(-10*e.mean().log10()),float(-10*e[:,mask].mean().log10())])
means=np.mean(values,0);progress=json.loads((BASE/'refined-progress.json').read_text());expected=next(x for x in progress if x['step']==d['step']);delta=max(abs(means[0]-expected['validation_psnr']),abs(means[1]-expected['validation_foreground_psnr']));assert delta<1e-4,delta
torch.save(d,p);audit={'passed':True,'bug':'Appearance-only optimizer regularized cached boundary parameters toward zero; cached validation geometry itself was unchanged.','fix':'Restore the exact joint-checkpoint controls used to build the appearance-stage cache; exclude frozen controls from optimizer.','direct_live_validation_psnr':float(means[0]),'direct_live_validation_foreground_psnr':float(means[1]),'maximum_difference_from_cached_validation_db':float(delta),'test_images_read_for_repair':False,'selection_changed':False};(BASE/'freeze-reload-audit.json').write_text(json.dumps(audit,indent=2));print(audit)
