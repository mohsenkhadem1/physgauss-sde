"""Verify zero-update integration preserves the actual renderer on fitting/validation frames."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
configure(0);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from argparse import Namespace
from scene_PIDG import GaussianModel,DeformModel
from scene_PIDG.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov,fov2focal
from scripts.pidg_dense_mechanics.field import DenseMechanics
run=ROOT/'runs/pidg-comparison-v01/full-pidg';meta=json.loads((run/'run.json').read_text());cfg={};exec((run/'config.py').read_text(),cfg);g=GaussianModel(3);g.load_ply(str(run/'model/point_cloud/iteration_50000/point_cloud.ply'),og_number_points=100000);d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(run/'model'),50000);m=DenseMechanics();m.initialize(d,g);frames=[]
for split in ['train','test']:
 doc=json.loads((ROOT/f'runs/pidg-comparison-v01/data/balls_reaction/fit/transforms_{split}.json').read_text());frames+=doc['frames']
pipe=Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);background=torch.zeros(3,device='cuda');rows=[]
for f in frames:
 mat=np.linalg.inv(np.array(f['transform_matrix']));R=-mat[:3,:3].T;R[:,0]=-R[:,0];T=-mat[:3,3];fx=doc['camera_angle_x'];fy=focal2fov(fov2focal(fx,320),180);cam=Camera(f['file_path'],R,T,fx,fy,torch.zeros(3,180,320),None,f['file_path'],0,fid=f['time']);t=cam.fid.reshape(1,1).expand(len(g.get_xyz),1)
 with torch.no_grad():
  p=d.step(g.get_xyz,t);before=render(cam,g,pipe,background,p['d_xyz'],p['d_rotation'],p['d_scaling'],False)['render'].clone()
 # Computing mechanics is deliberately allowed to build gradients but must not mutate appearance.
 loss=m(d,g,cam.fid)
 with torch.no_grad():
  p=d.step(g.get_xyz,t);after=render(cam,g,pipe,background,p['d_xyz'],p['d_rotation'],p['d_scaling'],False)['render']
 error=float((before-after).abs().max());assert error==0.;rows.append({'frame':Path(f['file_path']).name,'maximum_pixel_difference':error})
r={'passed':True,'frames':len(rows),'splits':'All 101 fitting and 11 validation frames; no test images read','max_pixel_difference':max(x['maximum_pixel_difference'] for x in rows),'rows':rows,'meaning':'Adding the mechanics computation preserves the native dense renderer exactly before optimization, across the entire fitting/validation interval.'};(ROOT/'reports/pidg-dense-mechanics-v01/identity-render-audit.json').write_text(json.dumps(r,indent=2));print('PASS',len(rows),'frames, exact pixel identity')
