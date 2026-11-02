"""Matched initial author-GS fit; dynamic transport/evaluation remains separate."""
import os,sys,json,pickle,subprocess,time,signal,importlib.util
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import native_appearance_pilot as helper
os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT=helper.ROOT;AUTHOR=helper.AUTHOR;PYTHON=helper.PYTHON
BASE=ROOT/'runs/compositional-dynamics-v17/matched-appearance-native-resolution'
REPORT=ROOT/'reports/compositional-dynamics-v17/matched-appearance-native-resolution'
DESIGN=ROOT/'reports/compositional-dynamics-v17/matched-appearance-native-resolution-design.json'
PLAN=REPORT/'plan.json'
sha=helper.sha;write=helper.write;read=helper.read

def projection():
    # Same tested author camera adapter; only prospective public camera catalog differs.
    spec=importlib.util.spec_from_file_location('_matched_camera_projection',Path(helper.__file__));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    os.environ['CUDA_VISIBLE_DEVICES']='1';m.REPORT=REPORT;REPORT.mkdir(parents=True,exist_ok=True)
    m.camera_specs=lambda:read(DESIGN)['cameras'];m.projection()
    p=REPORT/'appearance-camera-projection-smoke.json';x=read(p);x['design_sha256']=sha(DESIGN);x['adapter_sha256']=sha(__file__);write(p,x)

def prepare():
    design=read(DESIGN);projection=read(REPORT/'appearance-camera-projection-smoke.json')
    if projection['design_sha256']!=sha(DESIGN)or projection['adapter_sha256']!=sha(__file__)or not projection['passed']:raise ValueError('Pinned camera qualification')
    if PLAN.exists()or BASE.exists():raise ValueError('Preserve prepared inputs')
    sources=helper.inventory();sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    test=ROOT/'tests/test_matched_appearance_native_resolution_v17.py';sources[str(test.relative_to(ROOT))]=sha(test)
    plan=dict(design=design,design_sha256=sha(DESIGN),source_sha256=sources,projection_sha256=sha(REPORT/'appearance-camera-projection-smoke.json'),iterations=10000,training_cap_seconds=1800,physical_gpu=1)
    plan['command']=[str(PYTHON),str(AUTHOR/'gs_train.py'),'-s',str(BASE/'train-input'),'-m',str(BASE/'model'),'--iterations','10000','--resolution','1','--disable_viewer','--use_masks','--isotropic','--gs_init_opt','pcd','--lambda_depth','.001','--lambda_seg','1','--lambda_normal','0','--lambda_anisotropic','0','--sh_degree','3','--save_iterations','10000','--test_iterations','10000']
    REPORT.mkdir(parents=True,exist_ok=True);write(PLAN,plan)
    folder=BASE/'train-input';folder.mkdir(parents=True)
    from PIL import Image
    Ks=[];poses=[]
    for i,spec in enumerate([c for c in design['cameras']if c['role']=='train']):
        frame=helper.reference(spec)
        Image.fromarray(np.uint8(np.round(frame['rgb']*255))).save(folder/f'{i}.png')
        Image.fromarray(frame['body_visible'].astype('uint8')*255).save(folder/f'mask_{i}.png')
        np.save(folder/f'{i}_depth.npy',np.where(frame['body_visible'],frame['body_visible_depth']*1000,0).astype(np.float32))
        K,pose=helper.calibration(spec);Ks.append(K);poses.append(pose)
    with (folder/'camera_meta.pkl').open('wb')as f:pickle.dump(dict(intrinsics=Ks,c2ws=poses),f)
    with (folder/'interp_poses.pkl').open('wb')as f:pickle.dump([helper.calibration(c)[1]for c in design['cameras']if c['role']=='held'],f)
    xyz=np.unique(helper.surface().reshape(-1,3),axis=0)
    with (folder/'observation.ply').open('w')as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for x in xyz:f.write(' '.join(map(str,x))+' 127 127 127\n')
    camera_report=reader_projection(folder,design['cameras'][:3]);write(REPORT/'reader-projection.json',camera_report)
    plan['reader_projection_sha256']=sha(REPORT/'reader-projection.json');write(PLAN,plan)
    write(BASE/'input-manifest.json',dict(plan_sha256=sha(PLAN),files={str(p.relative_to(ROOT)):sha(p)for p in sorted(folder.iterdir())},initial_points=len(xyz),train_views=3,held_images_generated=False))

def validate_camera(view,spec):
    K,_=helper.calibration(spec)
    if (view.image_width,view.image_height)!=(spec['width'],spec['height']):raise ValueError('Native reader/image dimensions mismatch')
    if not np.allclose(np.asarray(view.K),K,atol=1e-8,rtol=0):raise ValueError('Native reader intrinsic mismatch')

def reader_projection(folder,cameras):
    import tempfile,shutil
    from types import SimpleNamespace
    sys.path.insert(0,str(AUTHOR))
    import torch
    from gaussian_splatting.scene.dataset_readers import readQQTTSceneInfo
    from gaussian_splatting.utils.camera_utils import loadCam
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    rows=[]
    with tempfile.TemporaryDirectory(prefix='v17-reader-smoke-')as d:
        copied=Path(d)/'inputs';shutil.copytree(folder,copied)
        scene=readQQTTSceneInfo(str(copied),None,None,False,False,use_masks=True,gs_init_opt='pcd')
        if len(scene.train_cameras)!=3:raise ValueError('Native reader training census')
        for i,(info,spec)in enumerate(zip(scene.train_cameras,cameras)):
            view=loadCam(SimpleNamespace(resolution=1,data_device='cuda',train_test_exp=False),i,info,1.,False,False)
            validate_camera(view,spec)
            from PIL import Image
            expected=np.asarray(Image.open(copied/f'{i}.png'))/255.
            if not np.allclose(view.original_image.cpu().numpy().transpose(1,2,0),expected,atol=1e-7,rtol=0):raise ValueError('Reader unexpectedly resampled image')
            eye,R=helper.camera(spec).pose()
            for xyz in [[.04,0,0],[.08,.01,.01],[.06,-.01,-.01]]:
                xyz=np.asarray(xyz);q=R@(xyz-eye);expected=q[:2]/q[2]*spec['focal']+[spec['width']/2,spec['height']/2]
                pc=GaussianModel(0);pc._xyz=torch.tensor(xyz[None],dtype=torch.float32,device='cuda');pc._scaling=torch.full((1,3),float(np.log(.0007)),device='cuda');pc._rotation=torch.tensor([[1.,0,0,0]],device='cuda');pc._opacity=torch.full((1,1),5.,device='cuda')
                alpha=render(view,pc,None,torch.zeros(3,device='cuda'),override_color=torch.ones((1,3),device='cuda'))['render'][3].detach().cpu().numpy()
                yy,xx=np.indices(alpha.shape);actual=np.array([(alpha*(xx+.5)).sum(),(alpha*(yy+.5)).sum()])/alpha.sum();error=float(np.linalg.norm(actual-expected))
                if error>1.:raise ValueError('Native reader projection exceeds1pixel')
                rows.append(dict(camera=spec['name'],error_pixels=error))
    return dict(passed=True,rows=rows,max_error_pixels=max(r['error_pixels']for r in rows),scope='Actual author QQTT reader plus loadCam plus native rasterizer; exact image pixel identity and K/dimension checks. Temporary input copy prevents reader cache writes to pinned inputs.')

def guard():
    p=read(PLAN)
    if p['design_sha256']!=sha(DESIGN)or p['projection_sha256']!=sha(REPORT/'appearance-camera-projection-smoke.json'):raise ValueError('Design/projection changed')
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned source changed '+name)
    if p['reader_projection_sha256']!=sha(REPORT/'reader-projection.json')or not read(REPORT/'reader-projection.json')['passed']:raise ValueError('Native reader qualification changed')
    inputs=read(BASE/'input-manifest.json')
    if inputs['plan_sha256']!=sha(PLAN):raise ValueError('Input plan changed')
    for name,digest in inputs['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Training input changed '+name)
    if {p.name for p in (BASE/'train-input').glob('*.png')}!={f'{i}.png'for i in range(3)}|{f'mask_{i}.png'for i in range(3)}:raise ValueError('Exactly three training images/masks')
    with (BASE/'train-input/camera_meta.pkl').open('rb')as f:c=pickle.load(f)
    if len(c['c2ws'])!=3 or len(c['intrinsics'])!=3:raise ValueError('Exactly three camera records')
    return p

def train():
    p=guard();review=read(REPORT/'prefit-independent-review.json')
    if not review['passed']or review['plan_sha256']!=sha(PLAN)or review['input_manifest_sha256']!=sha(BASE/'input-manifest.json'):raise ValueError('Exact independent review required')
    with (BASE/'training-start.json').open('x')as f:json.dump(dict(plan_sha256=sha(PLAN),review_sha256=sha(REPORT/'prefit-independent-review.json')),f)
    started=time.monotonic();result=dict(complete=False,plan_sha256=sha(PLAN));env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYNPUT_BACKEND='dummy',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        with (BASE/'training.log').open('w')as log:
            proc=subprocess.Popen(p['command'],cwd=AUTHOR,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=p['training_cap_seconds'])
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                time.sleep(2)
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait();raise
        if code:raise RuntimeError('Native trainer returned '+str(code))
        cp=BASE/'model/point_cloud/iteration_10000/point_cloud.ply';guard()
        from plyfile import PlyData
        vertices=PlyData.read(str(cp))['vertex'].data
        if len(vertices)==0 or any(not np.isfinite(vertices[name]).all()for name in vertices.dtype.names):raise ValueError('Empty or nonfinite learned appearance checkpoint')
        result.update(complete=True,checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp),gaussian_count=len(vertices))
    except Exception as e:result['error']=repr(e);raise
    finally:result['wall_seconds']=time.monotonic()-started;write(BASE/'training-complete.json',result)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['projection','prepare','train']);a=p.parse_args();globals()[a.stage]()
