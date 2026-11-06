"""Complete-cohort summary and fixed-frame comparison panels."""
from pathlib import Path
import sys,json,subprocess,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/pidg-solver-forecast-v01';BASE=ROOT/'runs/pidg-solver-forecast-v01';OUT=ROOT/'output/pidg-solver-forecast-v01';OUT.mkdir(exist_ok=True)
fontpath=next(p for p in (Path.home()/'.local/share/fonts/arial').glob('*.TTF') if p.name.lower()=='arial.ttf');font=ImageFont.truetype(str(fontpath),18);small=ImageFont.truetype(str(fontpath),14);methods=['native-refined','hybrid','solver','constant-velocity'];names=['Native PIDG','Previous hybrid','Solver-driven','Constant velocity'];summary=[];panels=[];manifest=[]
for scene in ['balls_reaction','rubber_duck','motion_kuro']:
 doc=json.loads((REPORT/scene/'results.json').read_text());motion=json.loads((REPORT/scene/'motion.json').read_text());base=BASE/scene/'evaluation';frames=sorted([p.stem for p in (base/'reference').glob('*.png') if '-mask' not in p.name],key=int);frame=frames[len(frames)//2];mask=np.asarray(Image.open(base/'reference'/(frame+'-mask.png')))>0;y,x=np.where(mask);box=(max(0,x.min()-10),max(0,y.min()-10),min(320,x.max()+11),min(180,y.max()+11));panel=Image.new('RGB',(1600,260),'white');draw=ImageDraw.Draw(panel);draw.text((8,3),scene.replace('_',' ').title()+' | middle declared future test frame '+frame,font=font,fill='black')
 for col,(method,label) in enumerate(zip(['reference']+methods,['Reference']+names)):
  draw.text((col*320+8,30),label,font=font,fill='black');path=base/method/(frame+'.png')
  if path.exists():
   im=Image.open(path).crop(box);im.thumbnail((312,170));panel.paste(im,(col*320+(320-im.width)//2,58+(170-im.height)//2))
  else:draw.text((col*320+8,110),'Rollout failed',font=font,fill='black')
  if method!='reference':
   row=next((r for r in doc['rows'] if r['method']==method and r['frame']==frame),None)
   if row:draw.text((col*320+8,235),f"Frame FG PSNR: {row['foreground_psnr_db']:.2f} dB",font=small,fill='black')
 panel.save(OUT/(scene+'-comparison.png'));panels.append(panel);manifest.append(dict(scene=scene,frame=frame,reference_crop=[int(v) for v in box],selection='Middle declared test frame; common reference-mask crop.'))
 for method in methods:
  row=dict(scene=scene,method=method,**doc['summary'][method]);row['flow_disagreement_px']=motion['summary'].get(method);summary.append(row)
 # Sparse recorded frames, displayed at three frames/s, with original frame IDs.
 proc=subprocess.Popen([str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1600x230','-r','3','-i','-','-an','-vcodec','libx264','-pix_fmt','yuv420p','-crf','18',str(OUT/(scene+'.mp4'))],stdin=subprocess.PIPE)
 for frame in frames:
  im=Image.new('RGB',(1600,230),'white');draw=ImageDraw.Draw(im)
  for col,(method,label) in enumerate(zip(['reference']+methods,['Reference']+names)):
   draw.text((col*320+8,4),label,font=font,fill='black');path=base/method/(frame+'.png')
   if path.exists():im.paste(Image.open(path),(col*320,27))
   else:draw.text((col*320+8,100),'Rollout failed',font=font,fill='black')
  draw.text((8,208),f'{scene} | recorded test frame {frame} | sparse-frame comparison, not real-time playback',font=small,fill='black');proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0
combined=Image.new('RGB',(1600,780),'white')
for i,p in enumerate(panels):combined.paste(p,(0,i*260))
combined.save(OUT/'three-scenes.png')
(REPORT/'summary.json').write_text(json.dumps(dict(rows=summary,figure_manifest=manifest,scope='Three scenes, shared chronological observation budget. No SI physical error labels. All failures retained; no aggregate advantage inferred from selected screenshots.'),indent=2));print(json.dumps(summary,indent=2))
