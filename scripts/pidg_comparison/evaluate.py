"""Matched saved-checkpoint evaluation; validation first, test only after selection."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--render-width',type=int,default=0);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--gpu',type=int,choices=[0,1,2],default=0);a=p.parse_args();configure(a.gpu)
    sys.path.insert(0,str(EXTERNAL/'upstream'))
    import torch,numpy as np
    from PIL import Image
    from argparse import Namespace
    from scene_PIDG import GaussianModel,DeformModel
    from scene_PIDG.cameras import Camera
    from gaussian_renderer import render
    from utils.graphics_utils import focal2fov,fov2focal
    from utils.loss_utils import ssim
    import lpips
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    from scripts.pidg_comparison.mechanics import deformation_gradient
    from utils.general_utils import quaternion_to_matrix
    run=Path(a.run);meta=json.loads((run/'run.json').read_text());assert meta['status']=='training_complete'
    if a.split=='test':
        selection=json.loads((run.parent/'selection.json').read_text());assert str(run.resolve()) in selection['test_runs']
    steps=meta['arguments']['steps'];width=a.render_width or meta['arguments']['width'];scene=meta['arguments']['scene'];data=ROOT/'runs/pidg-comparison-v01/data'/scene
    doc=json.loads((data/('fit/transforms_test.json' if a.split=='validation' else 'evaluation/transforms_test.json')).read_text())
    cfg={};exec((run/'config.py').read_text(),cfg)
    g=GaussianModel(3);g.load_ply(str(run/'model/point_cloud'/f'iteration_{steps}'/'point_cloud.ply'),og_number_points=100000)
    d=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(run/'model'),steps)
    bg=torch.zeros(3,device='cuda');pipe=Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    out=run/(a.split if not a.render_width else f'{a.split}-render-{width}');out.mkdir(exist_ok=True);rows=[]
    with torch.no_grad():
        for f in doc['frames']:
            name=Path(f['file_path']).name;im=Image.open(data/'released'/(f['file_path']+'.png')).convert('RGBA');size=(width,round(im.height*width/im.width));arr=np.asarray(im.resize(size,Image.Resampling.LANCZOS)).copy()/255.
            gt=torch.tensor(np.floor(arr[:,:,:3]*arr[:,:,3:4]*255)/255,dtype=torch.float32,device='cuda').permute(2,0,1);mask=torch.tensor(arr[:,:,3]>.5,device='cuda');matrix=np.linalg.inv(np.array(f['transform_matrix']));R=-matrix[:3,:3].T;R[:,0]=-R[:,0];T=-matrix[:3,3]
            fx=doc['camera_angle_x'];fy=focal2fov(fov2focal(fx,size[0]),size[1]);cam=Camera(f['file_path'],R,T,fx,fy,gt,None,name,0,fid=f['time'])
            package=d.step(g.get_xyz,cam.fid.reshape(1,1).expand(len(g.get_xyz),1));pred=render(cam,g,pipe,bg,package['d_xyz'],package['d_rotation'],package['d_scaling'],False)['render'].clamp(0,1)
            err=(pred-gt).square();row={'frame':name,'time':f['time'],'psnr_db':float(-10*err.mean().clamp_min(1e-12).log10()),'psnr_channel_mean_db':float((-10*err.flatten(1).mean(1).clamp_min(1e-12).log10()).mean()),'lpips_alex':float(perceptual(pred[None]*2-1,gt[None]*2-1).mean()),'ssim':float(ssim(pred,gt)),'foreground_rmse':float(err[:,mask].mean().sqrt()),'foreground_psnr_db':float(-10*err[:,mask].mean().clamp_min(1e-12).log10())}
            ids=torch.linspace(0,len(g.get_xyz)-1,min(256,len(g.get_xyz)),device='cuda').long();x=g.get_xyz[ids]
            def mapping(q):
                delta=d.step(q,cam.fid.reshape(1,1).expand(len(q),1))['d_xyz'];return (quaternion_to_matrix(delta[:,:4])@q[...,None]).squeeze(-1)+delta[:,4:]
            J=torch.linalg.det(deformation_gradient(mapping,x));row.update(sampled_inversion_fraction=float((J<=0).float().mean()),sampled_abs_volume_change=float((J-1).abs().mean()))
            if not all(np.isfinite(value) for key,value in row.items() if key!='frame'):raise RuntimeError(f'Nonfinite metrics for frame {name}')
            np.savez_compressed(out/f'{name}.npz',prediction=pred.permute(1,2,0).cpu().numpy(),truth=gt.permute(1,2,0).cpu().numpy(),mask=mask.cpu().numpy())
            rows.append(row)
    metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['frame','time']}
    result={'split':a.split,'render_width':width,'training_width':meta['arguments']['width'],'method':meta['label'],'frame_count':len(rows),'metrics':metrics,'rows':rows,'limitations':'Volume change is a diagnostic, not ground-truth physics accuracy; no true states/forces released in this input. Frame interpolation, not changed-action forecasting.','checkpoint':steps,'run_metadata_sha256':hashlib.sha256((run/'run.json').read_bytes()).hexdigest()}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
