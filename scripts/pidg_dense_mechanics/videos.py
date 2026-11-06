"""Recorded held frames beside saved native/hybrid renders; no frame synthesis."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/pidg-dense-mechanics-v01';OUT=ROOT/'output/pidg-dense-mechanics-v01';OUT.mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--scene',default='all');args=p.parse_args();scenes=['balls_reaction','rubber_duck','motion_kuro'] if args.scene=='all' else [args.scene];font=ImageFont.truetype(str(Path.home()/'.local/share/fonts/arial/Arial.TTF'),22);bold=ImageFont.truetype(str(Path.home()/'.local/share/fonts/arial/Arialbd.TTF'),25)
for scene in scenes:
 r=json.loads((REPORT/scene/'results.json').read_text());selected=r['selection']['selected'];native=r['metrics']['native-refined']['640'];ours=r['metrics'][selected]['640'];folder=ROOT/'runs/pidg-dense-mechanics-v01'/scene;frames=OUT/(scene+'-frames');frames.mkdir(exist_ok=True)
 for i,(n,o) in enumerate(zip(native['rows'],ours['rows'])):
  assert n['frame']==o['frame'];a=np.load(folder/f'native-refined/test-render-640/{n["frame"]}.npz');b=np.load(folder/f'{selected}/test-render-640/{o["frame"]}.npz');canvas=Image.new('RGB',(1920,460),'white');draw=ImageDraw.Draw(canvas)
  for j,(label,im) in enumerate(zip(['Reference','PIDG','Ours: dense hybrid'],[a['truth'],a['prediction'],b['prediction']])):
   canvas.paste(Image.fromarray(np.round(im.clip(0,1)*255).astype('uint8')),(j*640,45));draw.text((j*640+12,10),label,font=bold,fill='black')
  draw.text((12,413),f'{scene.replace("_"," ").title()} | frame {n["frame"]} | held-frame reconstruction, not a new-action forecast',font=font,fill='black');canvas.save(frames/f'{i:04d}.png')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-framerate','12','-i',str(frames/'%04d.png'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(scene+'.mp4'))],check=True)
 print(scene,len(native['rows']),'frames',flush=True)
