"""Sealed forecast evaluation. All target reads happen after physical fitting."""
import argparse,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from PIL import Image,ImageDraw,ImageFont
 from types import SimpleNamespace
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
 from scripts.solver_driven_gaussians.score_motion import metrics
 from scripts.mechanics_gaussian_bridge.bridge import render
 from utils.loss_utils import ssim
 import subprocess
 torch.set_num_threads(2);r=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v02';out=ROOT/'reports/solver-driven-gaussians-v02';art=ROOT/'output/solver-driven-gaussians-v02';art.mkdir(exist_ok=True,parents=True)
 font=ImageFont.truetype('/home/bayespc2/.local/share/fonts/arial/Arial.ttf',18) if Path('/home/bayespc2/.local/share/fonts/arial/Arial.ttf').exists() else ImageFont.truetype(str(next(Path('/home/bayespc2/.local/share/fonts/arial').glob('*.TTF'))),18)
 labels=['Reference','Ours: learned mechanics','Analytic mechanics'];rows=[];geom=[]
 pairappearance=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,get_opacity=r.appearance.get_opacity.repeat(2,1),get_features=r.appearance.get_features.repeat(2,1,1))
 def image(x,view):
  if x.ndim==2:return r.image(x,view)
  states=[r.state(xx) for xx in x];state={k:torch.cat([s[k] for s in states]) for k in states[0]};return render(r.cameras[view],pairappearance,state,r.background)['render'].clamp(0,1)
 def pil(im):return Image.fromarray((im.detach().cpu().numpy().transpose(1,2,0)*255).round().astype('uint8'))
 for case in ['calibration','side_poke','early_release','force_pulse','higher_drop','pair_drop']:
  truth=np.load(base/('reference-pair' if case=='pair_drop' else 'reference')/(case+'.npz'));preds=[np.load(base/f/(case+'.npz')) for f in ['final-learned','final-analytic']];times=truth['times'];keep=times>.14
  for method,d in zip(['learned','analytic'],preds):
   assert np.allclose(times,d['times']);meta=json.loads((base/('final-'+method)/(case+'.json')).read_text())
   if case!='pair_drop':assert not meta['target_states_read'] and hashlib.sha256((base/('final-'+method)/(case+'.npz')).read_bytes()).hexdigest()==meta['prediction_sha256']
   if case=='pair_drop':
    ms=[metrics(d['x'][keep,j],truth['x'][keep,j],r.model.rest.cpu().numpy()) for j in range(2)];met={k:float(np.mean([m[k] for m in ms])) for k in ms[0]}
   else:met=metrics(d['x'][keep],truth['x'][keep],r.model.rest.cpu().numpy())
   geom.append(dict(case=case,method=method,**met,minimum_J=meta['minimum_J'],penetration_mm=1000*meta.get('max_pair_penetration_m',meta.get('max_penetration_m',0))))
  frames=range(0,len(times),4);video=subprocess.Popen([str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x290','-r','25','-i','-','-an','-vcodec','libx264','-pix_fmt','yuv420p','-crf','18',str(art/(case+'.mp4'))],stdin=subprocess.PIPE)
  for frame in frames:
   ims={}
   with torch.no_grad():
    for view in range(4):
     target=image(truth['x'][frame],view);mask=(target-.96).abs().max(0).values>.03
     if view==2:ims[0]=pil(target)
     for j,d in enumerate(preds):
      pred=image(d['x'][frame],view);union=mask|((pred-.96).abs().max(0).values>.03);mse=(pred-target).square()
      rows.append(dict(case=case,method=['learned','analytic'][j],time_s=float(times[frame]),view=view,split='novel' if view>=2 else 'familiar',forecast=bool(times[frame]>.14),psnr=float(-10*mse.mean().clamp_min(1e-12).log10()),foreground_psnr=float(-10*mse[:,mask].mean().clamp_min(1e-12).log10()),union_psnr=float(-10*mse[:,union].mean().clamp_min(1e-12).log10()),ssim=float(ssim(pred,target))))
      if view==2:ims[j+1]=pil(pred)
   panel=Image.new('RGB',(960,290),'white');draw=ImageDraw.Draw(panel)
   for j,label in enumerate(labels):panel.paste(ims[j],(j*320,28));draw.text((j*320+8,5),label,font=font,fill='black')
   draw.text((8,268),f'{case.replace("_"," ")} | {times[frame]:.3f} s | unseen camera | 5x slow motion',font=font,fill='black');video.stdin.write(panel.tobytes())
   if abs(times[frame]-.24)<.002:panel.save(art/(case+'-comparison.png'))
  video.stdin.close();assert video.wait()==0
  print(case,[g for g in geom if g['case']==case],flush=True)
 summary=[]
 for case in sorted(set(x['case'] for x in rows)):
  for method in ['learned','analytic']:
   for split in ['familiar','novel']:
    rr=[x for x in rows if x['case']==case and x['method']==method and x['split']==split and x['forecast']];summary.append(dict(case=case,method=method,views=split,frames=len(rr),**{k:float(np.mean([x[k] for x in rr])) for k in ['psnr','foreground_psnr','union_psnr','ssim']}))
 result=dict(geometry=geom,render_summary=summary,render_frames=rows,scope='Controlled known-volume, known-density, fixed-appearance RGB identification. Independent Mooney-Rivlin reference, learned invariant law versus neo-Hookean control, equal proposal/refinement budgets. Forecasts after 0.14 s; views 2/3 never fitted. Synthetic shared-appearance render; not public-dataset PSNR. Pair copies inferred material to a second object. No friction/CCD.')
 (out/'final-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
