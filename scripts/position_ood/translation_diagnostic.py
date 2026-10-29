"""Separate imposed rigid translation from the induced change in object shape."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT
from scripts.position_ood.score import deformation
from scripts.phystwin.sequential_reference import write

def run(split):
 if 'test' in split.split('_'):
  from scripts.position_ood.seal import verify
  verify()
 protocol=json.loads((BASE/'transfer-protocol.json').read_text());rows=[];missing=[]
 for i in range(len(protocol[split])):
  p=BASE/f'fields/{split}-{i}.npz'
  if not p.exists():missing.append(i);continue
  with np.load(p) as z:f=z['field'];poses=z['poses']
  # Both tools have the same translation for actions >=3. Subtract this known
  # action, not a best fit against prediction, to isolate the environmental effect.
  translation=poses[3:,:3] if poses.ndim==2 else poses[3:,0,:3]
  # Controls are [left translation xyz, left rotation xyz, right translation xyz, right rotation xyz].
  delta=f[3:]-f[0]-translation[:,None,:]
  rows.append(dict(object=i,translation_induced_coordinate_rms_mm=(1000*np.sqrt(np.mean(delta**2,axis=(1,2)))).tolist(),deformation=deformation(f)))
 result=dict(split=split,rows=rows,missing=missing,scope='Reference geometry diagnostic. Change beyond the prescribed whole-object translation; includes field-induced rigid offsets and deformation. Nonrigid RMS additionally removes the best rigid transform.')
 write(OUT/f'{split}-translation-diagnostic.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',default='transfer_development');a=p.parse_args();run(a.split)
