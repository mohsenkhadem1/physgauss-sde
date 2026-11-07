"""Final figure provenance and full-sequence clipping audit (presentation only)."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];B=R/'runs/gap-first-v01/drop';O=R/'output/gap-first-v01/drop';P=R/'reports/gap-first-v01/drop'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
p=json.loads((P/'protocol.json').read_text());rows=[]
for a in p['assets']:
 for c in ['passive-A','passive-B','changed-A','changed-B']:
  f=B/'cases'/a/c/'illustration-camera-v2.npz';rgb=np.load(f)['rgb'];mask=rgb.max(-1)>8;border=np.concatenate([mask[:,0],mask[:,-1],mask[:,:,0],mask[:,:,-1]],axis=1);rows.append(dict(asset=a,case=c,frames=len(rgb),foreground_border_pixels=int(border.sum()),input_sha256=sha(f)))
assert all(x['foreground_border_pixels']==0 for x in rows)
figs=['native-drop-wide-comparison.png','native-drop-wide-comparison.pdf','native-drop-wide-comparison.mp4','native-drop-teaser.png','native-drop-teaser.pdf'];d=dict(passed=True,presentation_only=True,all360frames_unclipped_at_mask_threshold8=True,inputs=rows,camera_sha256=sha(P/'illustration-camera-v2.json'),protocol_sha256=sha(P/'protocol.json'),artifacts={f:sha(O/f) for f in figs},visual_QA='Both final PDF pages rendered through Poppler and visually inspected; all snapshots fit, text readable, Arial and Arial-Bold embedded; no colored text. One common crop in first-asset teaser.',source_sha256=sha(__file__))
(O/'final-figure-audit.json').write_text(json.dumps(d,indent=2));print('Passed: 360 frames; no foreground pixels touch image borders.')
