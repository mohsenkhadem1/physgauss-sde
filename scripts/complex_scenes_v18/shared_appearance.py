"""Fixed learned Gaussian appearance and identical author K16 motion transport.

Controlled object-only rendering: no appearance refitting, independent native
reconstruction, or environment-occlusion claim. Every method uses the same map.
"""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
# Reuse the previously qualified local CUDA toolchain without choosing a GPU.
TOOLCHAIN=Path('/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env')
os.environ.setdefault('CUDA_HOME',str(TOOLCHAIN));os.environ.setdefault('CPATH',str(TOOLCHAIN/'targets/x86_64-linux/include'))
os.environ.setdefault('CC',str(TOOLCHAIN/'bin/x86_64-conda-linux-gnu-gcc'));os.environ.setdefault('CXX',str(TOOLCHAIN/'bin/x86_64-conda-linux-gnu-c++'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.5');os.environ.setdefault('MAX_JOBS','2')
os.environ['PATH']=str(TOOLCHAIN/'bin')+os.pathsep+os.environ['PATH']
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.world_queries import Query
from scripts.physical_refinement_v15.observations import Camera

AUTHOR=ROOT/'runs/compositional-dynamics-v17/PhysTwin-author'
CHECKPOINT=ROOT/'runs/compositional-dynamics-v17/matched-appearance-native-resolution/model/point_cloud/iteration_10000/point_cloud.ply'


def camera_specs():
    return [dict(name=name,eye=list(eye),target=[.10,0.,-.01],width=512,height=384,focal=550.)for name,eye in [('front',(.25,-.42,.25)),('side',(-.22,-.22,.15)),('back',(.22,.30,.20))]]


def initial_pose(query,body):
    R=np.eye(3);t=np.zeros(3)
    if query.family=='container_drop':
        R=Rotation.from_euler('y',query.initial_rotation_deg,degrees=True).as_matrix();center=np.array([.06,0,0]);t=center-R@center+np.array([0,0,query.initial_height_m])
    if query.family=='deformable_pair'and body==1:t=np.array([.008,.006,.09]if query.variant else[.18,.004,.003])
    return R,t


def render(trajectory,query,output,frame_count=7):
    import torch
    from types import SimpleNamespace
    sys.path.insert(0,str(AUTHOR))
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render as native_render
    from gs_render import remove_gaussians_with_low_opacity
    from scripts.compositional_dynamics_v17.matched_gaussian_states import native_states
    trajectory=Path(trajectory);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from scripts.complex_scenes_v18.appearance_controls import controls
    times,positions=controls(trajectory.parent)
    if positions.ndim!=4 or positions.shape[2:]!=(125,3)or not np.isfinite(positions).all():raise ValueError('Finite common volumetric control grid required')
    count=positions.shape[1]
    if count!=(2 if query.family=='deformable_pair'else 1):raise ValueError('Public object identity mismatch')
    frames=np.unique(np.rint(np.linspace(0,len(times)-1,min(frame_count,len(times)))).astype(int))
    pc=GaussianModel(3);pc.isotropic=True;pc.load_ply(str(CHECKPOINT));pc=remove_gaussians_with_low_opacity(pc)
    histories=[];records=[]
    with torch.no_grad():
        for body in range(count):
            R,t=initial_pose(query,body);xyz=pc.get_xyz@torch.tensor(R.T,dtype=torch.float32,device='cuda')+torch.tensor(t,dtype=torch.float32,device='cuda')
            quat=pc.get_rotation.detach().cpu().numpy();rot=Rotation.from_matrix(R)*Rotation.from_quat(quat[:,[1,2,3,0]])
            quat=torch.tensor(rot.as_quat()[:,[3,0,1,2]],device='cuda',dtype=torch.float32)
            histories.append(native_states(positions[:,body],xyz,pc.get_features_dc.squeeze(1),quat,pc.get_opacity))
        current=copy.copy(pc)
        for attr in ('_features_dc','_features_rest','_opacity','_scaling'):setattr(current,attr,torch.cat([getattr(pc,attr)]*count))
        for frame in frames:
            current._xyz=torch.tensor(np.concatenate([h['centers_m'][frame]for h in histories]),device='cuda')
            current._rotation=torch.tensor(np.concatenate([h['quaternions_wxyz'][frame]for h in histories]),device='cuda')
            for spec in camera_specs():
                cam=Camera(tuple(spec['eye']),tuple(spec['target']),spec['width'],spec['height'],spec['focal']);eye,R=cam.pose()
                w2c=np.eye(4);w2c[:3,:3]=R;w2c[:3,3]=-R@eye;K=np.array([[cam.focal,0,cam.width/2],[0,cam.focal,cam.height/2],[0,0,1]])
                view=SimpleNamespace(K=torch.tensor(K,dtype=torch.float32,device='cuda'),image_width=cam.width,image_height=cam.height,world_view_transform=torch.tensor(w2c.T,dtype=torch.float32,device='cuda'))
                result=native_render(view,current,None,torch.full((3,),245/255,device='cuda'));rgba=result['render'].cpu().numpy();depth=result['depth'].cpu().numpy()
                if not np.isfinite(rgba).all()or not np.isfinite(depth).all():raise ValueError('Nonfinite native image')
                mask=rgba[3]>=.5;path=output/f'{frame:03d}-{spec["name"]}.npz'
                np.savez_compressed(path,rgb=np.clip(rgba[:3].transpose(1,2,0),0,1),body_visible=mask,body_visible_depth=np.where(mask,depth,np.inf),time_s=float(times[frame]))
                records.append(dict(frame=int(frame),time_s=float(times[frame]),camera=spec,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    from dataclasses import asdict
    state_seal=json.loads((trajectory.parent/'complete.json').read_text())
    report=dict(completed=True,scope='Object-only common learned appearance and author K16 transport of a common 5x5x5 volumetric grid; no refitting or native end-to-end appearance claim',rigid_environment_rendered=False,public_query=asdict(query),appearance_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),physical_state_sha256=state_seal.get('nodal_state_sha256',state_seal.get('particle_state_sha256')),gaussians_per_object=len(pc.get_xyz),records=records)
    write(output/'images-seal.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trajectory',required=True);p.add_argument('--query-json',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=7);a=p.parse_args()
    import torch
    torch.set_num_threads(1);r=render(a.trajectory,Query(**json.loads(Path(a.query_json).read_text())),a.output,a.frames);print(json.dumps(dict(images=len(r['records']),gaussians=r['gaussians_per_object'])))
