"""Prospective static author-GS optimization pilot, separate from inferred mechanics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
TOOLCHAIN='/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env'
os.environ.setdefault('CUDA_HOME',TOOLCHAIN)
os.environ.setdefault('CPATH',TOOLCHAIN+'/targets/x86_64-linux/include')
os.environ.setdefault('CC',TOOLCHAIN+'/bin/x86_64-conda-linux-gnu-gcc')
os.environ.setdefault('CXX',TOOLCHAIN+'/bin/x86_64-conda-linux-gnu-c++')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.5')
os.environ.setdefault('MAX_JOBS','2')
os.environ['PATH']='/home/bayespc2/vscode/guassianSDE/runs/compositional-dynamics-v17/render-env/bin:'+TOOLCHAIN+'/bin:'+os.environ['PATH']
import argparse,hashlib,json,pickle,subprocess,sys,time,signal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/compositional-dynamics-v17/appearance-pilot'
REPORT=ROOT/'reports/compositional-dynamics-v17'
AUTHOR=ROOT/'runs/compositional-dynamics-v17/PhysTwin-author'
PLAN=REPORT/'appearance-pilot-plan.json'
PYTHON=ROOT/'runs/compositional-dynamics-v17/render-env/bin/python'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())
def camera_specs():
    result=[]
    for i,angle in enumerate([0,90,180,270,45,225]):
        az=np.deg2rad(angle);el=np.deg2rad(25);target=np.array([.06,0,0]);eye=target+.32*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        result.append(dict(name=f'camera-{i}',role='train'if i<4 else'held',eye=eye.tolist(),target=target.tolist(),width=848,height=480,focal=700.))
    return result

def surface():
    u,v=np.meshgrid(np.linspace(0,1,17),np.linspace(0,1,17),indexing='ij');frames=[]
    for face in range(6):
        q=np.zeros((17,17,3));axis=face//2;q[...,axis]=face%2;q[...,[j for j in range(3)if j!=axis]]=np.stack([u,v],-1)
        frames.append(q*np.array([.12,.06,.06])+np.array([0,-.03,-.03]))
    return np.array(frames)

def camera(spec):
    from scripts.physical_refinement_v15.observations import Camera
    return Camera(tuple(spec['eye']),tuple(spec['target']),spec['width'],spec['height'],spec['focal'])

def calibration(spec):
    cam=camera(spec);eye,R=cam.pose();w2c=np.eye(4);w2c[:3,:3]=R;w2c[:3,3]=-R@eye
    K=np.array([[cam.focal,0,cam.width/2],[0,cam.focal,cam.height/2],[0,0,1.]])
    return K,np.linalg.inv(w2c)

def reference(spec):
    from scripts.compositional_dynamics_v17.render_sequence import render_frame
    frame=render_frame(surface(),camera(spec),time_s=0.);frame['rgb']=np.round(frame['rgb']*255)/255
    return frame

def inventory():
    paths=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/render_sequence.py',ROOT/'scripts/compositional_dynamics_v17/render_metrics.py',ROOT/'scripts/physical_refinement_v15/observations.py']
    paths+=sorted(AUTHOR.rglob('*.py'));paths+=[ROOT/'runs/compositional-dynamics-v17/render-environment-freeze.txt']
    return {str(p.relative_to(ROOT)):sha(p)for p in paths if '__pycache__'not in p.parts}

def command(plan):
    return [str(PYTHON),str(AUTHOR/'gs_train.py'),'-s',str(BASE/'train-input'),'-m',str(BASE/'model'),'--iterations',str(plan['iterations']),'--resolution','1','--disable_viewer','--use_masks','--isotropic','--gs_init_opt','pcd','--lambda_depth','.001','--lambda_seg','1','--lambda_normal','0','--lambda_anisotropic','0','--sh_degree','3','--save_iterations',str(plan['iterations']),'--test_iterations',str(plan['iterations'])]

def prepare():
    projection_report=read(REPORT/'appearance-camera-projection-smoke.json')
    if projection_report['passed']is not True or projection_report['source_sha256']!=sha(__file__):raise ValueError('Current native camera projection smoke required')
    if PLAN.exists()or BASE.exists():raise ValueError('Preserve prepared plan and data; no overwrite')
    plan=dict(scope='Static known-geometry author native GS appearance optimization; NOT full PhysTwin inverse/dynamic replication.',iterations=1000,optimizer_updates=999,seed=0,selection='Native final iteration save only; no heldout selection',training_cap_seconds=900,cameras=camera_specs(),geometry=dict(size_m=[.12,.06,.06],center_m=[.06,0,0],surface_grid=17),initialization='Known canonical surfacepoints with uniform RGB127/255; no hidden procedural texture colors',loss='Author gs_train, isotropic pcd, depth.001/seg1/normal0/anisotropic0; other author defaults',background='Native training black after alpha masking; evaluation public.96 background rounded to245/255 like reference',image_encoding='Training and evaluator RGB rounded to8bit; depth metres internally, QQTT input millimetres',mask_rule='Predicted alpha>=.5, joint optical-axis depth, rawRGB clipped[0,1] before metrics; clipping count retained',held_seal='No held RGB/depth generated before all6 predictions committed. Heldcamera poses are public.',source_sha256=inventory(),author_commit='81c718790a37e5e0102eb77af2c6edd34a9db25f',timing='Prospective before fitting; no held reference outcomes exist',runtime=str(PYTHON.relative_to(ROOT)))
    plan['camera_projection_sha256']=sha(REPORT/'appearance-camera-projection-smoke.json');plan['timeout_scope']='Whole trainer process group TERM then KILL, including compilerchildren';plan['command']=command(plan);write(PLAN,plan)
    folder=BASE/'train-input';folder.mkdir(parents=True)
    from PIL import Image
    Ks=[];poses=[]
    for i,spec in enumerate(plan['cameras'][:4]):
        f=reference(spec);Image.fromarray(np.uint8(np.round(f['rgb']*255))).save(folder/f'{i}.png');Image.fromarray(f['body_visible'].astype('uint8')*255).save(folder/f'mask_{i}.png')
        np.save(folder/f'{i}_depth.npy',np.where(f['body_visible'],f['body_visible_depth']*1000,0).astype(np.float32));K,pose=calibration(spec);Ks.append(K);poses.append(pose)
    with open(folder/'camera_meta.pkl','wb')as f:pickle.dump(dict(intrinsics=Ks,c2ws=poses),f)
    with open(folder/'interp_poses.pkl','wb')as f:pickle.dump([calibration(c)[1]for c in plan['cameras'][4:]],f)
    xyz=np.unique(surface().reshape(-1,3),axis=0)
    with open(folder/'observation.ply','w')as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for x in xyz:f.write(' '.join(map(str,x))+' 127 127 127\n')
    write(BASE/'input-manifest.json',dict(plan_sha256=sha(PLAN),files={str(p.relative_to(ROOT)):sha(p)for p in sorted(folder.iterdir())},initial_points=len(xyz)))
    return plan

def guard():
    p=read(PLAN)
    if 'camera_projection_sha256'in p and sha(REPORT/'appearance-camera-projection-smoke.json')!=p['camera_projection_sha256']:raise ValueError('Projection qualification changed')
    for path,h in p['source_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Source changed: '+path)
    m=read(BASE/'input-manifest.json')
    if m['plan_sha256']!=sha(PLAN):raise ValueError('Input plan mismatch')
    for path,h in m['files'].items():
        if sha(ROOT/path)!=h:raise ValueError('Training input changed: '+path)
    expected={f'{i}.png'for i in range(4)}|{f'mask_{i}.png'for i in range(4)}
    if {q.name for q in (BASE/'train-input').glob('*.png')}!=expected:raise ValueError('Exact4 training images/masks; no held inputs')
    with open(BASE/'train-input/camera_meta.pkl','rb')as f:metadata=pickle.load(f)
    if len(metadata['c2ws'])!=4 or len(metadata['intrinsics'])!=4:raise ValueError('Exactly4 training cameras')
    return p

def train():
    p=guard();review=read(REPORT/'appearance-pilot-prefit-independent-review.json')
    if review.get('passed')is not True or review.get('plan_sha256')!=sha(PLAN)or review.get('input_manifest_sha256')!=sha(BASE/'input-manifest.json'):raise ValueError('Exact prefit review required')
    start_path=BASE/'training-start.json'
    with start_path.open('x')as f:json.dump(dict(plan_sha256=sha(PLAN),review_sha256=sha(REPORT/'appearance-pilot-prefit-independent-review.json')),f)
    t=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYNPUT_BACKEND='dummy',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    status=dict(plan_sha256=sha(PLAN),complete=False)
    try:
        with open(BASE/'training.log','w')as log:
            proc=subprocess.Popen(p['command'],cwd=AUTHOR,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=p['training_cap_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        status['returncode']=code
        if code:raise RuntimeError('Native trainer failed; see retained log')
        ck=BASE/f"model/point_cloud/iteration_{p['iterations']}/point_cloud.ply"
        status.update(complete=True,checkpoint=str(ck.relative_to(ROOT)),checkpoint_sha256=sha(ck));guard()
    except Exception as e:status.update(complete=False,error=repr(e))
    status['wall_seconds']=time.monotonic()-t;write(BASE/'training-complete.json',status)
    if not status['complete']:raise RuntimeError(status)

def predict():
    p=guard();status=read(BASE/'training-complete.json')
    if not status['complete']or status['plan_sha256']!=sha(PLAN)or sha(ROOT/status['checkpoint'])!=status['checkpoint_sha256']:raise ValueError('Trained checkpoint mismatch')
    if(BASE/'prediction-commit.json').exists()or(BASE/'held-reference').exists():raise ValueError('Preserve committed predictions; no posttarget prediction rerun')
    sys.path.insert(0,str(AUTHOR));import torch
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    pc=GaussianModel(3);pc.isotropic=True;pc.load_ply(str(ROOT/status['checkpoint']))
    records=[]
    with torch.no_grad():
        for spec in p['cameras']:
            K,c2w=calibration(spec);native=SimpleNamespace(K=torch.tensor(K,dtype=torch.float32,device='cuda'),image_width=spec['width'],image_height=spec['height'],world_view_transform=torch.tensor(np.linalg.inv(c2w).T,dtype=torch.float32,device='cuda'))
            result=render(native,pc,SimpleNamespace(),torch.full((3,),245/255,device='cuda'),use_gsplat=True)
            rgba=result['render'].cpu().numpy();depth=result['depth'].cpu().numpy();assert np.isfinite(rgba).all()and np.isfinite(depth).all()
            rgb=rgba[:3].transpose(1,2,0);mask=rgba[3]>=.5;path=BASE/'predictions'/f"{spec['name']}.npz";path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,rgb=np.clip(rgb,0,1),raw_rgb=rgb,alpha=rgba[3],body_visible=mask,body_visible_depth=np.where(mask,depth,np.inf))
            records.append(dict(name=spec['name'],role=spec['role'],path=str(path.relative_to(ROOT)),sha256=sha(path),clipped_channels=int(((rgb<0)|(rgb>1)).sum())))
    guard();write(BASE/'prediction-commit.json',dict(plan_sha256=sha(PLAN),training_complete_sha256=sha(BASE/'training-complete.json'),records=records))

def score():
    p=guard();commit=read(BASE/'prediction-commit.json')
    if commit['plan_sha256']!=sha(PLAN)or commit['training_complete_sha256']!=sha(BASE/'training-complete.json'):raise ValueError('Prediction binding changed')
    if [(r['name'],r['role'])for r in commit['records']]!=[(s['name'],s['role'])for s in p['cameras']]:raise ValueError('Exact6 predictions required')
    for r in commit['records']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Prediction changed')
    from scripts.compositional_dynamics_v17.render_metrics import compare_renderings
    rows=[];tiles=[]
    from PIL import Image,ImageDraw
    for s,r in zip(p['cameras'],commit['records']):
        f=reference(s);refpath=BASE/'held-reference'/f"{s['name']}.npz";refpath.parent.mkdir(exist_ok=True)
        if refpath.exists():raise ValueError('Preserve reference/scoring artifacts')
        np.savez_compressed(refpath,**{k:v for k,v in f.items()if k!='camera'})
        with np.load(ROOT/r['path'])as a:pred={k:a[k]for k in ['rgb','body_visible','body_visible_depth']}
        pred['camera']=f['camera'];pred['time_s']=0.;metrics=compare_renderings(pred,f);rows.append(dict(name=s['name'],role=s['role'],metrics=metrics,reference_sha256=sha(refpath),prediction_sha256=r['sha256']))
        tile=Image.new('RGB',(848,270),'white');draw=ImageDraw.Draw(tile);draw.text((5,3),f"{s['name']} {s['role']} | actual procedural RGB / native fitted GS | PSNR {metrics['psnr_foreground_union_db']:.2f}dB union",fill='black')
        for j,img in enumerate([f['rgb'],pred['rgb']]):tile.paste(Image.fromarray(np.uint8(np.round(img*255))).resize((424,240)),(j*424,30))
        tiles.append(tile)
    gallery=Image.new('RGB',(1696,810),'white')
    for i,tile in enumerate(tiles):gallery.paste(tile,((i%2)*848,(i//2)*270))
    gallery_path=REPORT/'appearance-pilot-comparison.png';gallery.save(gallery_path)
    report=dict(scope=p['scope'],gallery_sha256=sha(gallery_path),plan_sha256=sha(PLAN),prediction_commit_sha256=sha(BASE/'prediction-commit.json'),rows=rows,summary=summarize(rows))
    write(REPORT/'appearance-pilot-results.json',report)

def summarize(rows):
    result={}
    for role in ['train','held']:
        cohort=[r for r in rows if r['role']==role];result[role]={'count':len(cohort),'metrics':{}}
        for key in ['psnr_global_db','psnr_foreground_union_db','ssim_rgb_global','ssim_rgb_foreground_union','silhouette_iou','depth_joint_rmse_m']:
            values=[r['metrics'][key]for r in cohort if r['metrics'][key]is not None]
            result[role]['metrics'][key]=dict(mean=float(np.mean(values))if values else None,available=len(values),expected=len(cohort))
    return result

def projection():
    sys.path.insert(0,str(AUTHOR));import torch
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    rows=[]
    for spec in camera_specs():
        K,c2w=calibration(spec);eye,R=camera(spec).pose()
        for xyz in [[.04,0,0],[.08,.01,.01],[.06,-.01,-.01]]:
            xyz=np.asarray(xyz);q=R@(xyz-eye);expected=q[:2]/q[2]*spec['focal']+[spec['width']/2,spec['height']/2]
            pc=GaussianModel(0);pc._xyz=torch.tensor(xyz[None],dtype=torch.float32,device='cuda');pc._scaling=torch.full((1,3),float(np.log(.0007)),device='cuda');pc._rotation=torch.tensor([[1.,0,0,0]],device='cuda');pc._opacity=torch.full((1,1),5.,device='cuda')
            view=SimpleNamespace(K=torch.tensor(K,dtype=torch.float32,device='cuda'),world_view_transform=torch.tensor(np.linalg.inv(c2w).T,dtype=torch.float32,device='cuda'),image_width=spec['width'],image_height=spec['height'])
            alpha=render(view,pc,None,torch.zeros(3,device='cuda'),override_color=torch.ones((1,3),device='cuda'))['render'][3].detach().cpu().numpy()
            yy,xx=np.indices(alpha.shape);measured=np.array([(alpha*(xx+.5)).sum(),(alpha*(yy+.5)).sum()])/alpha.sum();error=float(np.linalg.norm(measured-expected));assert error<=1.
            rows.append(dict(camera=spec['name'],xyz_m=xyz.tolist(),expected_pixel_center=expected.tolist(),measured_pixel_center=measured.tolist(),error_pixels=error))
    write(REPORT/'appearance-camera-projection-smoke.json',dict(passed=True,max_error_pixels=max(r['error_pixels']for r in rows),rows=rows,scope='Public calibration only: isolated artificialGaussians, no block heldappearance/reference reads',source_sha256=sha(__file__)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','train','predict','score','projection']);stage=a.parse_args().stage;globals()[stage]()
