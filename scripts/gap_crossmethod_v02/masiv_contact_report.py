"""Fixed-appearance native mechanics ambiguity and calibrated-force control.
No fitted MASIV checkpoint, no population calibration/generalization claim.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from masiv_render_diagnostic import renderer,render_frames,ROOT
DATA=ROOT/'runs/gap-crossmethod-v02/masiv/contact-v02'
OUT=ROOT/'output/gap-crossmethod-v02/masiv-contact'
OUT.mkdir(parents=True,exist_ok=True)

def rms(a,b):return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
def aligned_rms(a,b):
 a=a-a.mean(0);b=b-b.mean(0);u,_,vt=np.linalg.svd(a.T@b);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
 return rms(a@u@d@vt,b)
def centroid(im):
 w=np.linalg.norm(1-im,axis=-1);y,x=np.indices(w.shape)
 return np.array([(w*x).sum()/w.sum(),(w*y).sum()/w.sum()])

def main():
 render=renderer(); rows=[]; all_images={};trials=[]
 for shape in ['torus','ellipsoid']:
  images={};state={}
  for scale in [1,2]:
   for force,axis in [(0,'x'),(10,'x'),(20,'z')]:
    state[scale,force]=np.load(DATA/f'{shape}-scale{scale}-gaussian{force}{axis}.npz')['x']
    images[scale,force]=render_frames(state[scale,force],state[scale,force][0],render)
  all_images[shape]=images
  row={'shape':shape}
  for force in [0,10,20]:
   a,b=images[1,force],images[2,force];mask=np.any(np.minimum(a,b)<.99,axis=-1)
   row[str(force)]=dict(full_psnr_db=float(-10*np.log10(max(np.mean((a-b)**2),1e-15))),
    union_foreground_psnr_db=float(-10*np.log10(max(np.mean((a-b)[mask]**2),1e-15))),
    final_point_RMS_mm=1000*rms(state[1,force][-1],state[2,force][-1]),
    rigid_aligned_final_difference_mm=1000*aligned_rms(state[1,force][-1],state[2,force][-1]),
    world1_deformation_from_initial_mm=1000*aligned_rms(state[1,force][0],state[1,force][-1]))
  cal=np.stack([centroid(images[s,10][-1]) for s in [1,2]])
  target=np.stack([state[s,20][-1] for s in [1,2]])
  rng=np.random.default_rng(713)
  for truth in [0,1]:
   for rep in range(100):
    obs=cal[truth]+rng.normal(0,.5,2)
    logw=-.5*np.sum(((obs-cal)/.5)**2,axis=-1);w=np.exp(logw-logw.max());w/=w.sum();picked=int(w.argmax())
    prior_errors=np.array([rms(target[0],target[j]) for j in [0,1]])
    post_errors=np.array([rms(target[picked],target[j]) for j in [0,1]])
    trials.append(dict(shape=shape,truth=truth,repeat=rep,posterior_world2=float(w[1]),
     prior_risk=float(np.mean(prior_errors>.003)),posterior_risk=float(w@(post_errors>.003)),
     prior_error_mm=float(prior_errors[truth]*1000),posterior_error_mm=float(post_errors[truth]*1000),
     same_evidence_map_error_mm=float(post_errors[truth]*1000)))
  row['calibration_image_centroid_gap_px']=float(np.linalg.norm(cal[1]-cal[0]));rows.append(row)
 report=dict(classification='native MASIV mechanics plus supplied rescaled pretrained law and shared fixed Gaussian appearance; NOT fitted end-to-end MASIV',
  hypothesis_scope='two constructed density/material scale hypotheses, truth included, identical fixed appearance',
  measurement='final calibration rendered image intensity-foreground centroid, independent Gaussian noise sigma0.5pixels per axis',
  action='10N localized x-force calibration, then separate-reset20N localized z-force held target; neither force chosen using truth',
  tolerance_mm=3,rows=rows,trials=trials,
  summary=dict(prior_mean_error_mm=float(np.mean([r['prior_error_mm'] for r in trials])),
   posterior_mean_error_mm=float(np.mean([r['posterior_error_mm'] for r in trials])),
   prior_failure_rate=float(np.mean([r['prior_error_mm']>3 for r in trials])),
   posterior_failure_rate=float(np.mean([r['posterior_error_mm']>3 for r in trials])),
   same_evidence_baseline='identical MAP; no acquisition superiority claimed'))
 (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
 font=ImageFont.truetype(str(next(Path('/home/bayespc2/.local/share/fonts/arial').glob('*.TTF'))),20)
 labels=['Passive A','Passive B','New force A','New force B','RGB difference x10']
 mask=np.zeros((384,384),dtype=bool)
 for images in all_images.values():
  for im in images.values(): mask|=np.any(im<.99,axis=(0,3))
 yy,xx=np.where(mask);x0,x1=max(0,int(xx.min())-12),min(384,int(xx.max())+13)
 y0,y1=max(0,int(yy.min())-12),min(384,int(yy.max())+13)
 width,height=x1-x0,y1-y0
 for t in range(31):
  canvas=Image.new('RGB',(5*width,2*(height+65)+65),'white');d=ImageDraw.Draw(canvas)
  d.text((10,5),'Identical passive appearance does not fix the response to a new force',fill='black',font=font)
  for r,shape in enumerate(['torus','ellipsoid']):
   for c,(s,f) in enumerate([(1,0),(2,0),(1,20),(2,20)]):
    y=35+r*(height+65);im=Image.fromarray(np.uint8(np.clip(all_images[shape][s,f][t],0,1)*255)).crop((x0,y0,x1,y1))
    canvas.paste(im,(width*c,y+24));d.text((width*c+8,y),labels[c],fill='black',font=font)
   diff=np.clip(np.max(np.abs(all_images[shape][1,20][t]-all_images[shape][2,20][t]),axis=-1)*10,0,1)
   rgb=np.stack([np.ones_like(diff),1-diff,1-diff],axis=-1)
   im=Image.fromarray(np.uint8(rgb*255)).crop((x0,y0,x1,y1))
   canvas.paste(im,(width*4,y+24));d.text((width*4+8,y),labels[4],fill='black',font=font)
   d.text((10,y+height+27),f"{shape}: final new-force response separation {rows[r]['20']['final_point_RMS_mm']:.2f} mm",fill='black',font=font)
  d.text((10,canvas.height-25),'Native mechanics + fixed shared appearance; not fitted MASIV rendering scores.',fill='black',font=font)
  canvas.save(OUT/f'frame-{t:03d}.png')
 Image.open(OUT/'frame-030.png').save(OUT/'comparison.png')
 print(json.dumps(dict(rows=rows,summary=report['summary']),indent=2))
if __name__=='__main__':main()
