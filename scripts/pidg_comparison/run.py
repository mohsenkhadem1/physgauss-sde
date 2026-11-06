"""Run pinned PIDG with shared compatibility fixes and optional frozen energy.
Upstream remains external and unmodified. All runtime edits are recorded.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXTERNAL=Path.home()/'.local/share/physgauss-external/pidg-official-v01'
def configure(gpu):
    env=EXTERNAL/'env'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_HOME=str(env),CC=str(env/'bin/x86_64-conda-linux-gnu-gcc'),CXX=str(env/'bin/x86_64-conda-linux-gnu-g++'),TORCH_CUDA_ARCH_LIST='7.5',MAX_JOBS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(EXTERNAL/'cache'),MPLCONFIGDIR=str(EXTERNAL/'mpl-config'))
    includes=[env/'targets/x86_64-linux/include',*sorted((env/'lib/python3.10/site-packages/nvidia').glob('*/include'))]
    os.environ['CPATH']=os.environ['CPLUS_INCLUDE_PATH']=':'.join(map(str,includes));os.environ['PATH']=str(env/'bin')+':'+os.environ['PATH']
def main():
    p=argparse.ArgumentParser();p.add_argument('--dense-weight',type=float,default=0);p.add_argument('--warm-from',default='');p.add_argument('--gpu',type=int,choices=[0,1,2],required=True);p.add_argument('--weight',type=float,default=0);p.add_argument('--steps',type=int,default=5000);p.add_argument('--width',type=int,default=320);p.add_argument('--output',required=True);p.add_argument('--scene',default='balls_reaction');p.add_argument('--flow',action='store_true');a=p.parse_args();configure(a.gpu)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXTERNAL/'upstream'))
    import torch,numpy as np
    import gaussian_renderer
    from scene_PIDG import dataset_readers
    # Correct rectangular image intrinsics for BOTH methods.
    source=Path(dataset_readers.__file__).read_text();needle='FovY, FovX = fovx, fovy';assert source.count(needle)==1
    exec(compile(source.replace(needle,'FovX, FovY = fovx, fovy'),dataset_readers.__file__,'exec'),dataset_readers.__dict__)
    import utils.camera_utils as camera_utils
    camera_source=Path(camera_utils.__file__).read_text()
    camera_source=camera_source.replace('mask_resized=cam_info.mask_image','mask_resized=None if cam_info.mask_image is None else cam_info.mask_image.transpose(2,0,1)')
    exec(compile(camera_source,camera_utils.__file__,'exec'),camera_utils.__dict__)
    original_render=gaussian_renderer.render
    def compatible_render(*args,**kwargs):
        if len(args)<8 and 'enable_flow_grad' not in kwargs:kwargs['enable_flow_grad']=False
        return original_render(*args,**kwargs)
    gaussian_renderer.render=compatible_render
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    # Each run owns its initial point file; identical seed and data access.
    data=ROOT/'runs/pidg-comparison-v01/data'/a.scene/('fit320' if a.flow else 'fit')
    if a.flow and not (data/'preprocessing.json').exists():raise RuntimeError('Training-only flow preprocessing must finish first')
    local=out/'input';local.mkdir(exist_ok=True)
    for item in data.iterdir():
        if item.name=='points3d.ply':continue
        dst=local/item.name
        if not dst.exists():dst.symlink_to(item.resolve(),target_is_directory=item.is_dir())
    np.random.seed(0)
    if not (local/'points3d.ply').exists():
        from utils.sh_utils import SH2RGB
        dataset_readers.storePly(str(local/'points3d.ply'),np.random.random((100000,3))*2.6-1.3,SH2RGB(np.random.random((100000,3))/255.)*255.)
    source_path=EXTERNAL/'upstream/train_pidg.py';code=source_path.read_text()
    changes={'renderer_default_flow_flag':False,'camera_fov_assignment':'FoVx=fovx,FoVy=fovy','checkpoint_scalar':'float(cur_psnr) instead of cur_psnr.item()','input_time':'common physical frame index normalized to [0,1]','validation':'upstream test slot contains only 11 reserved train images','flow_enabled':a.flow,'unresized_mask_layout':'HWC to CHW; matches resized branch','gpu_image_loading':'eager; broken optional on-the-fly path not used'}
    code=code.replace('cur_psnr.item()','float(cur_psnr)')
    if a.warm_from:
        from scene_PIDG import Scene,DeformModel
        warm=Path(a.warm_from).resolve();warm_meta=json.loads((warm/'run.json').read_text());assert warm_meta['status']=='training_complete'
        warm_step=warm_meta['arguments']['steps'];scene_init=Scene.__init__;deform_setup=DeformModel.train_setting
        def warm_scene(self,*args,**kwargs):
            scene_init(self,*args,**kwargs)
            self.gaussians.load_ply(str(warm/'model/point_cloud'/f'iteration_{warm_step}'/'point_cloud.ply'),og_number_points=100000)
            self.gaussians.max_radii2D=torch.zeros(len(self.gaussians.get_xyz),device='cuda')
        def warm_deform(self,opt):
            self.load_weights(str(warm/'model'),warm_step);deform_setup(self,opt)
        Scene.__init__=warm_scene;DeformModel.train_setting=warm_deform
        changes['warm_restart']={'from':str(warm),'iteration':warm_step,'optimizer':'fresh, identical schedules across arms','velocity_and_stress_net':'reinitialized in all arms; upstream checkpoints omit these weights','densification':'disabled; preserve each model final geometry population'}
    prior=None
    if a.weight or a.dense_weight:
        from scripts.pidg_comparison.mechanics import ConstitutivePrior,CHECKPOINT
        if a.dense_weight:
            from scripts.pidg_dense_mechanics.field import DenseMechanics
            prior=DenseMechanics()
        else:prior=ConstitutivePrior()
        needle='        loss.backward()';assert code.count(needle)==1
        code=code.replace(needle,'        if iteration > opt.warm_up:\n            loss = loss + mechanics_weight * mechanics_prior(deform, gaussians, fid)\n'+needle)
        if a.dense_weight:
            code=code.replace('        loss.backward()', '        loss.backward()\n        mechanics_prior.step(mechanics_weight)')
            changes['dense_mechanics']='Frozen bulk law + 16 spatial Gaussian load/attachment fields, 16 temporal knots; dense renderer retained'
        changes['constitutive_checkpoint_sha256']=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    cfg=out/'config.py';cfg.write_text((EXTERNAL/'upstream/arguments/pidg'/f'{dict(rubber_duck="rubber_dack").get(a.scene,a.scene)}.py').read_text()+f'\nuse_flow = {a.flow!r}\nsave_image_dir = {str(out / "flow_debug")!r}\n')
    if a.warm_from:
        with cfg.open('a') as f:f.write(f'\nwarm_up=0\ndensify_until_iter=0\nposition_lr_init=1.6e-6\nposition_lr_final=1.6e-7\nposition_lr_max_steps={a.steps}\ndeform_lr_max_steps={a.steps}\nfeature_lr=0.0005\nopacity_lr=0.005\nscaling_lr=0.0001\nrotation_lr=0.0001\n')
    metadata={'status':'running','started':time.time(),'arguments':vars(a),'upstream_commit':'b77fd2107a50c875d7219bb4f82388d5e5ea1319','runtime_changes':changes,'label':('Dense renderer + shared material and Gaussian boundary mechanics' if a.dense_weight else 'PIDG + frozen shared material energy (integration ablation)' if a.weight else 'PIDG compatible reproduction')+('' if a.flow else ' — without optical flow'),'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}}
    if a.dense_weight:metadata['source_sha256'].update({str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'scripts/pidg_dense_mechanics').glob('*.py')})
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    sys.argv=[str(source_path),'-s',str(local),'-m',str(out/'model'),'--conf',str(cfg),'--iterations',str(a.steps),'--resolution',str(a.width),'--test_iterations',str(a.steps),'--save_iterations',str(a.steps)]
    try:
        exec(compile(code,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path),'mechanics_prior':prior,'mechanics_weight':a.dense_weight or a.weight})
        if a.dense_weight:prior.save(out/'dense-mechanics.pt');metadata['dense_mechanics_last']=prior.last
        metadata['status']='training_complete'
    except BaseException as e:
        metadata.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:
        metadata['finished']=time.time();(out/'run.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
