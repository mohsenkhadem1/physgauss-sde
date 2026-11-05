"""Matched measured-image metrics and explicitly qualitative new-view comparisons."""
from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];run=ROOT/'runs/pidg-mechanics-v04';out=ROOT/'reports/pidg-mechanics-v01';out.mkdir(exist_ok=True);r=json.loads((run/'evaluation/results.json').read_text());font=ImageFont.truetype(str(Path.home()/'.local/share/fonts/arial/Arial.TTF'),22)
methods=[('PIDG',ROOT/'runs/pidg-view-audit-v03/refined-pidg'),('Material prior',ROOT/'runs/pidg-view-audit-v03/refined-ours-00001'),('Ours (mechanics)',run/'evaluation')]
# Every predeclared novel camera/time is shown; no outcome-based image selection.
canvas=Image.new('RGB',(2240,9*205+55),'white');draw=ImageDraw.Draw(canvas);draw.text((10,5),'Same cameras and times. New angles have no reference RGB: qualitative comparison only.',font=font,fill='black')
for k,row in enumerate(r['novel_views']):
 for mi,(label,folder) in enumerate(methods):
  x=(k%7)*320;y=55+((k//7)*3+mi)*205;im=Image.open(folder/row['image']).resize((320,180),Image.Resampling.LANCZOS);canvas.paste(im,(x,y));draw.text((x+3,y+181),f"{label} | {row['yaw_deg']:+d} deg",font=font,fill='black')
canvas.save(out/'new-camera-comparison.png')
# Three uniformly spaced released test views, not selected for gain.
fixed=[r['novel_views'][i*7+3]['frame'] for i in range(3)];panel=Image.new('RGB',(1920,3*395+40),'white');d=ImageDraw.Draw(panel)
for i,f in enumerate(fixed):
 baseline=np.load(ROOT/f'runs/pidg-comparison-v02/refined-pidg/test-render-640/{f:04d}.npz');truth=Image.fromarray((baseline['truth']*255).round().astype('uint8'));pidg=Image.fromarray((baseline['prediction']*255).round().astype('uint8'));ours=Image.open(run/f'evaluation/{f:04d}.png')
 for j,(label,im) in enumerate(zip(['Reference','PIDG','Connected mechanics'],[truth,pidg,ours])):
  panel.paste(im,(j*640,40+i*395));d.text((j*640+8,15+i*395),f'{label} | frame {f}',font=font,fill='black')
panel.save(out/'held-out-comparison.png')
comparison={}
for label,folder in [('PIDG','refined-pidg'),('Material-only','refined-ours-00001')]:
 comparison[label]={}
 for w in [320,640,1600]:
  suffix='test' if w==320 else f'test-render-{w}';doc=json.loads((ROOT/'runs/pidg-comparison-v02'/folder/suffix/'metrics.json').read_text());comparison[label][str(w)]=doc['metrics']
comparison['Connected mechanics']=r['means'];candidates={}
for version in [1,2,3,4]:
 p=ROOT/f'runs/pidg-mechanics-v{version:02d}';candidates[p.name]={name:json.loads((p/name).read_text()) for name in ['appearance-progress.json','joint-progress.json','joint-progress-80.json'] if (p/name).exists()}
audit={'completed':True,'comparison':comparison,'validation_development':candidates,'evaluation':r,'scope':'Existing paper arbitrary-mesh equilibrium adapter with train-inferred Gaussian attachment targets, not the coupled transient Q2 benchmark adapter. Six-object spherical geometry initialization is an added explicit prior.','limitations':['Training reconstruction supplies the initial radius/depth gauge; subsequent silhouettes infer position targets.','No measured forces/materials or camera-OOD reference images; no physical-accuracy or novel-view PSNR win is established.','Inferred controls use the full training sequence; temporal interpolation is not future-action forecasting.','No coupled inter-object collision solve in this equilibrium adapter; GPU solver convergence is not evidence of true contact recovery.'],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['scripts/pidg_mechanics','scripts/mechanics_gaussian_bridge'] for p in (ROOT/folder).glob('*.py')}}
(out/'comparison-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(comparison,indent=2))
# Preserve compact input, fitting and selection provenance; raw data stay local.
for name in ['input-protocol.json','mechanics-fit.json','mechanics-seal.json','appearance-selection.json','joint-selection.json','final-selection.json']:
 (out/name).write_bytes((run/name).read_bytes())
