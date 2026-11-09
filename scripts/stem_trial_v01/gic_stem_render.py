"""GIC held-camera (camera 0) renders for the stem trial (protocol_v02 part A).
Usage (cwd = GIC vendor dir, gic-env): gic_stem_render.py MODEL FIT_TRAJ.npz OUT_DIR [LABEL TRAJ.npz SCENE_DIR]...
 - deformation-network render of the model's own fitting episode on camera 0 (label 'deformnet-fit')
 - native fixed-particle Gaussians (train_gs_with_fixed_pcd) trained on cameras 1-10 of the fitting
   episode, driven by GIC's simulated fitting trajectory (label 'physics-fit') and by each given
   trajectory against camera 0 of the given scene.
Saves uint8 renders; scoring is done separately from the PNG references."""
import os, sys, json
from pathlib import Path
from argparse import Namespace, ArgumentParser
R = Path(__file__).resolve().parents[2]
V = R/'runs/gap-crossmethod-v02/vendor/gic'
sys.path.insert(0, str(V))
import numpy as np, torch
from scene import Scene, GaussianModel, DeformModel, dataset_readers
from scene.dataset_readers import readCamerasFromAllData
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from new_trajectory import load_pcd_file
from arguments import OptimizationParams

original = dataset_readers.readPACNeRFInfo
def partition(*a, **k):
    info = original(*a, **k)
    if not k.get('read_cam', True):
        return info
    return info._replace(train_cameras=[c for c in info.train_cameras if c.uid != 0],
                         test_cameras=[c for c in info.test_cameras if c.uid == 0])
dataset_readers.sceneLoadTypeCallbacks['PacNeRF'] = partition


def cam0(scene_dir, cfg):
    infos = [c for c in readCamerasFromAllData(str(scene_dir), cfg.white_background) if c.uid == 0]
    infos = sorted(infos, key=lambda c: c.fid)
    return cameraList_from_camInfos(infos, 1.0, cfg)


def to_u8(t):
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)


def main():
    model, fit_traj, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); extra = sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    cfg = eval((model/'cfg_args').read_text(), {'Namespace': Namespace})
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([1., 1, 1] if cfg.white_background else [0., 0, 0], device='cuda')
    phys = json.loads((model/'0-pred.json').read_text())
    with torch.no_grad():
        g = GaussianModel(cfg.sh_degree); sc = Scene(cfg, g, load_iteration=40000, shuffle=False)
        deform = DeformModel(cfg); deform.load_weights(str(model), 40000)
        imgs = []
        for view in sorted(sc.getTestCameras(), key=lambda v: float(v.fid)):
            view.load2device(); d = deform.step(g.get_xyz, view.fid.reshape(1, 1))
            imgs.append(to_u8(render(view, g, pipe, bg, *d, False)['render'])); view.load2device('cpu')
        np.savez_compressed(out/'deformnet-fit.npz', images=np.stack(imgs))
        del g, sc, deform; torch.cuda.empty_cache()
    vol = load_pcd_file(str(model), 40000)
    xyz = torch.tensor(np.load(fit_traj)['xyz'], device='cuda')
    assert torch.allclose(xyz[0], vol), 'fit trajectory must start at the reconstructed particles'
    parser = ArgumentParser(); op = OptimizationParams(parser); opt = op.extract(parser.parse_args(['--reg_scale', '--reg_alpha']))
    marker = model/'stem-fixpcd-trained.json'  # outside GIC's checkpoint folder (its scanner parses names)
    # No in-loop evaluation: GIC's final-iteration report stacks every training image on the GPU
    # before the checkpoint is saved, which exhausts 11 GB with 370 frames.
    scene = train_gs_with_fixed_pcd(vol, cfg, opt, pipe, [], [opt.iterations], [x-vol for x in xyz],
                                    phys['fps'], force_train=not marker.exists(), grid_size=phys['density_grid_size'])
    marker.write_text(json.dumps(dict(iterations=opt.iterations, fit_traj=str(fit_traj), frames=len(xyz))))
    jobs = [('physics-fit', fit_traj, Path(cfg.source_path))]+[(extra[i], Path(extra[i+1]), Path(extra[i+2])) for i in range(0, len(extra), 3)]
    with torch.no_grad():
        for label, traj, scene_dir in jobs:
            t = torch.tensor(np.load(traj)['xyz'], device='cuda')
            assert torch.allclose(t[0], vol, atol=1e-6), label
            views = cam0(scene_dir, cfg); assert len(views) == len(t), (label, len(views), len(t))
            imgs = []
            for f, view in enumerate(views):
                view.load2device()
                imgs.append(to_u8(render(view, scene.gaussians, pipe, bg, t[f]-vol, 0., 0., False)['render'])); view.load2device('cpu')
            np.savez_compressed(out/f'{label}.npz', images=np.stack(imgs)); print(label, len(imgs), flush=True)


if __name__ == '__main__':
    main()
