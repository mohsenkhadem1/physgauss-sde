"""Image-motion proxies computed only after sealed prediction scoring."""
import argparse,json
from pathlib import Path
import numpy as np,cv2
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];base=root/'runs/pidg-solver-forecast-v01'/a.scene/'evaluation';doc=json.loads((base/'results.json').read_text());cv2.setNumThreads(2);rows=[]
for method in doc['summary']:
 frames=sorted([r['frame'] for r in doc['rows'] if r['method']==method],key=int)
 for first,second in zip(frames[:-1],frames[1:]):
  gt0=cv2.imread(str(base/'reference'/(first+'.png')),0);gt1=cv2.imread(str(base/'reference'/(second+'.png')),0);p0=cv2.imread(str(base/method/(first+'.png')),0);p1=cv2.imread(str(base/method/(second+'.png')),0);mask=cv2.imread(str(base/'reference'/(first+'-mask.png')),0)>0
  flow=cv2.optflow.DualTVL1OpticalFlow_create();target=flow.calc(gt0,gt1,None);pred=flow.calc(p0,p1,None);err=np.linalg.norm(pred-target,axis=-1);rows.append(dict(method=method,first=first,second=second,foreground_flow_disagreement_px=float(err[mask].mean()),foreground_pixels=int(mask.sum())))
result=dict(scene=a.scene,summary={m:float(np.mean([r['foreground_flow_disagreement_px'] for r in rows if r['method']==m])) for m in doc['summary'] if any(r['method']==m for r in rows)},rows=rows,scope='TV-L1 disagreement between rendered and observed consecutive held frames, under matched cameras and frame gaps. Optical flow itself is estimated, not ground truth; occlusion/camera motion and texture errors contribute.')
(root/'reports/pidg-solver-forecast-v01'/a.scene/'motion.json').write_text(json.dumps(result,indent=2));print(result['summary'])
