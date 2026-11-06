"""Generic log-spaced physical proposals scored on calibration RGB, never labels.

Fallback for poorly observable interior deformation in reconstructed tracks.
The bank spans material scale, two attachment faces, attachment and tool strength.
"""
import argparse,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);p.add_argument('--kind',default='learned',choices=['learned','analytic']);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import numpy as np,torch
 from PIL import Image
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
 torch.set_num_threads(2);r=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v02';out=base/('proposal-bank-'+a.kind);out.mkdir(exist_ok=True)
 targets={(frame,view):torch.tensor(np.array(Image.open(base/'images'/f'{frame:04d}-view{view}.png'))/255.,device='cuda',dtype=torch.float32).permute(2,0,1) for frame in [40,70] for view in [0,1]}
 history=[];best=None;start=time.time()
 for material in [.25,1.,4.]:
  for face in [0,1]:
   for total_attachment in [100.,1000.,10000.]:
    for tool in [30.,300.,3000.]:
     theta=np.r_[material,np.zeros(8),tool,2.];theta[1+face*4:5+face*4]=total_attachment/4
     try:
      with torch.no_grad():
       trajectory=r.model.rollout(theta,kind=a.kind,duration=.14,dt=.0002);scores=[]
       for (frame,view),target in targets.items():
        pred=r.image(trajectory['x'][frame],view);mask=(target-.96).abs().max(0).values>.03;scores.append(float((pred-target).abs()[:,mask].mean()))
      record=dict(parameters=theta.tolist(),loss=float(np.mean(scores)),completed=True)
      if best is None or record['loss']<best['loss']:best=record
     except ValueError as error:record=dict(parameters=theta.tolist(),completed=False,error=str(error))
     history.append(record)
     if len(history)%9==0:print(a.kind,len(history),'best',best['loss'] if best else None,'seconds',time.time()-start,flush=True)
     (out/'status.json').write_text(json.dumps(dict(completed=False,best=best,trials=history),indent=2))
 if best is None:raise RuntimeError('No valid physical proposal')
 result=dict(completed=True,kind=a.kind,parameters=best['parameters'],trials=history,selection='Calibration fitting RGB at .08/.14 s, views0/1; not reserved validation frames or future targets.',future_positions_read=False,future_images_read=False,force_labels_read=False,material_labels_read=False,calibration_end_s=.14,scope='Generic log-spaced simulation proposal bank; iterative calibration initialization. Known geometry/density/cameras and commanded target path. No force/material labels used.')
 (out/'fit.json').write_text(json.dumps(result,indent=2));print(best,flush=True)
if __name__=='__main__':main()
