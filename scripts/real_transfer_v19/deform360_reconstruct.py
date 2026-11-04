"""Local, headless wrapper around the released Deform360 Splatfacto stage."""
import os
from pathlib import Path
TOOLCHAIN=Path('/home/bayespc2/.local/share/physgauss-external/pidg-official-v01/env')
os.environ.setdefault('CUDA_HOME',str(TOOLCHAIN))
os.environ.setdefault('CPATH',str(TOOLCHAIN/'targets/x86_64-linux/include'))
os.environ.setdefault('CC',str(TOOLCHAIN/'bin/x86_64-conda-linux-gnu-gcc'))
os.environ.setdefault('CXX',str(TOOLCHAIN/'bin/x86_64-conda-linux-gnu-c++'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.5');os.environ.setdefault('MAX_JOBS','2')
os.environ['PATH']=str(TOOLCHAIN/'bin')+':'+os.environ['PATH']
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,gc,json,time,hashlib

def main():
    p=argparse.ArgumentParser();p.add_argument('--object',default='046-sponge');p.add_argument('--episode',type=int,default=0)
    p.add_argument('--gpu',default='0');p.add_argument('--smoke',action='store_true');p.add_argument('--iterations',type=int,default=200)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    from deform360.processing.reconstruct_stage import (NerfstudioSplatTrainer,_hull_seed_for_frame,
        build_nerfstudio_dataset,process_reconstruction_episode)
    from deform360.processing.episode import load_episode_calibration
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2]/'runs/real-transfer-v19'
    aligned=root/'Deform360/processed'/a.object;episode=aligned/f'episode_{a.episode:04d}'
    class ClosingTrainer(NerfstudioSplatTrainer):
        def train(self,*args,**kwargs):
            # Nerfstudio keeps global writers across per-frame training calls.
            # Close them before upstream removes each frame's scratch directory.
            from nerfstudio.utils import writer
            try:
                return super().train(*args,**kwargs)
            finally:
                for event_writer in writer.EVENT_WRITERS:
                    tb=getattr(event_writer,'tb_writer',None)
                    if tb is not None:tb.close()
                writer.EVENT_WRITERS.clear()
                writer.EVENT_STORAGE.clear()
                gc.collect();torch.cuda.empty_cache()
    trainer=ClosingTrainer();trainer._method_configs['splatfacto'].vis='tensorboard'
    if a.smoke:
        cameras=['brics-odroid-001_cam0','brics-odroid-007_cam0','brics-odroid-013_cam0']
        calibration=load_episode_calibration(episode)
        intrinsics,extrinsics=calibration
        scratch=root/'reconstruction-smoke'/a.object
        scratch.mkdir(parents=True,exist_ok=False)
        seed=_hull_seed_for_frame(episode,0,scratch,cameras=cameras,intrinsics=intrinsics,
                                extrinsics=extrinsics,cube_half_extent_m=.5,voxel_resolution=100)
        build_nerfstudio_dataset(episode,0,scratch,cameras=cameras,intrinsics=intrinsics,extrinsics=extrinsics,seed_ply_path=seed)
        result=trainer.train(scratch,scratch,'smoke.ply',a.iterations)
        (scratch/'complete.json').write_text(json.dumps(dict(completed=True,iterations=a.iterations,cameras=cameras,
            output=str(result),sha256=hashlib.sha256(Path(result).read_bytes()).hexdigest()),indent=2)+'\n')
        print('SMOKE COMPLETE',result,flush=True)
    else:
        def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        quality_path=episode/'v19-view-quality.json'
        quality=json.loads(quality_path.read_text());accepted=sorted(quality['accepted_cameras'])
        if len(accepted)<8:raise ValueError('Initial multiview quality gate failed')
        # Freeze cameras before reconstruction: these images cannot supervise splats.
        held_views=accepted[::6]
        cameras=[c for c in accepted if c not in held_views]
        files=sorted(episode.glob('*/mask_refined.meta.json'))+sorted(episode.glob('*/mask_refined.h5'))+sorted(episode.glob('*/undistorted.mp4'))
        plan=dict(first_frame_iterations=1000,warm_start_iterations=a.iterations,
                  cameras=cameras,held_views=held_views,
                  view_split='Every sixth sorted initially qualified camera; initial masks used only for quality selection.',
                  view_quality_sha256=sha(quality_path),voxel_resolution=300,
                  input_sha256={str(p.relative_to(root)):sha(p) for p in files},
                  preprocessing='Grounding-DINO + SAM2, released Splatfacto stage; reduced iteration budget, development qualification required.')
        target=episode/'splatfacto';target.mkdir(exist_ok=True);path=target/'v19-plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Reconstruction inputs/budget changed')
        if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
        result=process_reconstruction_episode(aligned,a.episode,trainer=trainer,cameras=cameras,
                    first_frame_iterations=1000,warm_start_iterations=a.iterations,voxel_resolution=300,overwrite=False)
        print('Reconstruction complete',len(result),'frames',flush=True)

if __name__=='__main__':main()
