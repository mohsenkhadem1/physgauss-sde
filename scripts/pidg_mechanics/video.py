"""Labelled discrete camera sweep; no generated in-between views."""
from pathlib import Path
import json,subprocess
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[2];run=ROOT/'runs/pidg-mechanics-v04';report=ROOT/'reports/pidg-mechanics-v01';out=ROOT/'output/pidg-mechanics-v01';out.mkdir(exist_ok=True,parents=True);frames=run/'video-frames';frames.mkdir(exist_ok=True);r=json.loads((run/'evaluation/results.json').read_text());font=ImageFont.truetype(str(Path.home()/'.local/share/fonts/arial/Arial.TTF'),25)
methods=[('PIDG',ROOT/'runs/pidg-view-audit-v03/refined-pidg'),('Material-only integration',ROOT/'runs/pidg-view-audit-v03/refined-ours-00001'),('Ours: connected mechanics',run/'evaluation')]
for k,row in enumerate(r['novel_views']):
 im=Image.new('RGB',(1920,440),'white');draw=ImageDraw.Draw(im)
 for j,(label,p) in enumerate(methods):
  im.paste(Image.open(p/row['image']),(j*640,40));draw.text((j*640+10,7),label,font=font,fill='black')
 draw.text((10,407),f"t={row['time']:.3f}; camera yaw {row['yaw_deg']:+d} degrees. No reference image at new angles; qualitative only.",font=font,fill='black');im.save(frames/f'{k:03d}.png')
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-framerate','2','-i',str(frames/'%03d.png'),'-c:v','libx264','-crf','20','-r','24','-pix_fmt','yuv420p',str(out/'camera-comparison.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
# Readable selected-camera panel: fixed middle time, fixed -60/0/+60 angles.
canvas=Image.new('RGB',(1920,3*395+40),'white');d=ImageDraw.Draw(canvas)
for col,k in enumerate([8,10,12]):
 row=r['novel_views'][k]
 for mi,(label,p) in enumerate(methods):
  x=col*640;y=40+mi*395;canvas.paste(Image.open(p/row['image']),(x,y));d.text((x+8,y-30),f"{label} | {row['yaw_deg']:+d} degrees",font=font,fill='black')
canvas.save(report/'new-camera-preview.png')
