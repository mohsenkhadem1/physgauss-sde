import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(1);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch
from scripts.pidg_appearance_transfer.model import Transfer,BASE
from scripts.pidg_mechanics.appearance import camera_frame
from scripts.mechanics_gaussian_bridge.bridge import render
m=Transfer();doc=json.loads((ROOT/'runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_train.json').read_text());camera,gt,_,frame=camera_frame(doc['frames'][0],doc)
with torch.no_grad():
 state=m.raw(frame);expected=m.source;err=float((state['centers'].cpu()-expected['xyz']).abs().max());cov_err=float((state['covariance'].cpu()-expected['covariance']).abs().max());im=m.image(camera,frame);psnr=float(-10*(im-gt).square().mean().log10());assert err<1e-5 and abs(psnr-56.332)<.01
r={'passed':True,'mesh_binding_position_max_error':err,'mesh_binding_covariance_max_error':cov_err,'mesh_bound_reference_psnr':psnr,'gaussians':len(m.features)};(BASE/'mesh-binding-audit.json').write_text(json.dumps(r,indent=2));print(r)
