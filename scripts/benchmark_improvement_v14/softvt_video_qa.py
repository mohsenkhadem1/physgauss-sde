import json,subprocess,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
r=Path('reports/benchmark-improvement-v14/videos');binary=Path('runs/media-tooling/ffmpeg-7.0.2-amd64-static');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=json.loads((r/'provenance.json').read_text())['cases'];out=[]
for row in rows:
 p=Path(row['path']);assert sha(p)==row['sha256'];probe=json.loads(subprocess.check_output([str(binary/'ffprobe'),'-v','error','-count_frames','-show_streams','-of','json',str(p)]));s=probe['streams'][0];assert int(s['nb_read_frames'])==row['frames'];assert s['avg_frame_rate']=='20/1';assert (s['width'],s['height'])==(1500,500);assert abs(float(s['duration'])-row['frames']/20)<.001
 subprocess.run([str(binary/'ffmpeg'),'-v','error','-xerror','-i',str(p),'-f','null','-'],check=True)
 selected=[0,row['frames']//2,row['frames']-1];sheet=Image.new('RGB',(1500,3*525),'white');draw=ImageDraw.Draw(sheet)
 for j,n in enumerate(selected):
  fp=Path('/tmp')/(p.stem+f'-{n}.png');subprocess.run([str(binary/'ffmpeg'),'-v','error','-i',str(p),'-vf',f'select=eq(n\\,{n})','-frames:v','1','-y',str(fp)],check=True)
  with Image.open(fp)as im:sheet.paste(im,(0,j*525+25))
  draw.text((10,j*525+5),f'{p.stem} | suffix frame{n} / {row["frames"]} | original frame{row["frame_interval"][0]+n}',fill='black')
 target=r/(p.stem+'-contactsheet.png');sheet.save(target);out.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),frames=int(s['nb_read_frames']),fps=20,duration=float(s['duration']),dimensions=[1500,500],full_decode_pass=True,review_sheet=str(target),sheet_sha256=sha(target)))
(r/'machine-qa.json').write_text(json.dumps(dict(passed=True,visual_review_passed=False,visual_review_required=True,provenance_sha256=sha(r/'provenance.json'),clips=out),indent=2)+'\n');print(json.dumps(out,indent=2))
