"""Decode completed reporting clips and extract four settled-state review frames."""
import json
from pathlib import Path
import imageio_ffmpeg
from PIL import Image
from scripts.physical_refinement_v15.banks import ROOT,sha
from scripts.physical_refinement_v15.render_bank import write
REPORT=ROOT/'reports/physical-refinement-v15/videos'
def inspect(path):
 reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(reader);width,height=meta['size'];chosen={};count=0
 for i,frame in enumerate(reader):
  if i in (0,24,48,72):chosen[i]=Image.frombytes('RGB',(width,height),frame)
  count+=1
 if count!=96 or abs(meta['fps']-12)>1e-6 or abs(meta['duration']-8)>.05:raise ValueError('Unexpected video timeline')
 sheet=Image.new('RGB',(width,height*4),'white')
 for row,index in enumerate((0,24,48,72)):sheet.paste(chosen[index],(0,row*height))
 output=path.with_suffix('.contact-sheet.png');sheet.save(output)
 return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),decoded_frames=count,metadata=meta,contact_sheet=str(output.relative_to(ROOT)),contact_sheet_sha256=sha(output),visual_review='Required separately; machine decoding does not establish visual correctness')
def main():
 provenance=REPORT/'provenance.json';p=json.loads(provenance.read_text());rows=[]
 for entry in p['outputs']:
  path=ROOT/entry['path']
  if sha(path)!=entry['sha256']:raise ValueError('Video provenance mismatch')
  rows.append(inspect(path))
 write(REPORT/'machine-qa.json',dict(provenance_sha256=sha(provenance),source_sha256=sha(Path(__file__)),clips=rows))
if __name__=='__main__':main()
