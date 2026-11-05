"""Frozen-checkpoint, off-path camera sweep. No reference images or PSNR claims."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from PIL import Image
 from argparse import Namespace
 from scene_PIDG import GaussianModel,DeformModel
 from scene_PIDG.cameras import Camera
 from gaussian_renderer import render
 from utils.graphics_utils import focal2fov,fov2focal
 run=Path(a.run);meta=json.loads((run/'run.json').read_text());steps=meta['arguments']['steps'];cfg={};exec((run/'config.py').read_text(),cfg)
 g=GaussianModel(3);g.load_ply(str(run/'model/point_cloud'/f'iteration_{steps}'/'point_cloud.ply'),og_number_points=100000)
 d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(run/'model'),steps)
 base=ROOT/'runs/pidg-comparison-v01/data/balls_reaction';doc=json.loads((base/'evaluation/transforms_test.json').read_text());all_doc=json.loads((base/'released/transforms_all.json').read_text());allposes=np.array([f['transform_matrix'] for f in all_doc['frames']])
 out=ROOT/'runs/pidg-view-audit-v03'/run.name;out.mkdir(exist_ok=True,parents=True);rows=[]
 # Predeclared temporal quantiles and yaw offsets; no selection by appearance.
 frames=[doc['frames'][i] for i in [0,len(doc['frames'])//2,len(doc['frames'])-1]]
 with torch.no_grad():
  for f in frames:
   for angle in [-90,-60,-30,0,30,60,90]:
    rad=np.deg2rad(angle);rot=np.array([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    pose=np.array(f['transform_matrix']);pose[:3,:3]=rot@pose[:3,:3];pose[:3,3]=rot@pose[:3,3]
    mat=np.linalg.inv(pose);R=-mat[:3,:3].T;R[:,0]=-R[:,0];T=-mat[:3,3];w,h=640,360;fx=doc['camera_angle_x'];fy=focal2fov(fov2focal(fx,w),h)
    cam=Camera('novel',R,T,fx,fy,torch.zeros(3,h,w,device='cuda'),None,'novel',0,fid=f['time'])
    v=d.step(g.get_xyz,cam.fid.reshape(1,1).expand(len(g.get_xyz),1));im=render(cam,g,Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False),torch.zeros(3,device='cuda'),v['d_xyz'],v['d_rotation'],v['d_scaling'],False)['render'].clamp(0,1)
    name=f"{Path(f['file_path']).name}-yaw{angle:+04d}.png";Image.fromarray((im.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/name)
    rows.append({'image':name,'time':f['time'],'yaw_deg':angle,'camera_to_world':pose.tolist(),'nearest_released_camera_center_distance':float(np.linalg.norm(allposes[:,:3,3]-pose[:3,3],axis=1).min()),'ground_truth_available':angle==0})
 (out/'manifest.json').write_text(json.dumps({'run':str(run),'rotation':'World Z axis about world origin; camera orientation and position rotate together','selection':'First, middle, last test frames; fixed yaw offsets -90,-60,-30,0,30,60,90','claim':'Qualitative only. No ground truth for new poses; no PSNR or winner inferred.','rows':rows},indent=2))
if __name__=='__main__':main()
