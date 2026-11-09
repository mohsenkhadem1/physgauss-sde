"""Native PAC-NeRF (b613048) on the stem trial.
fit  SCENE OUTDIR [E_init nu_init]  : static -> velocity -> dynamic stages on cameras 1-10, all frames
forecast CKPT_DIR OUT.npz NFRAMES   : replay the fitted model (native render_sequence) and save particles
                                      and camera-0 renders; force from STEM_FORCE_JSON via the labelled adapter.
Density is fixed by config (1000), as in PAC-NeRF's released configs; only E, nu and velocity are fitted."""
import os, sys, json, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
WORK = R/'runs/gap-crossmethod-v02/pacnerf-work'
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(R/'runs/gap-crossmethod-v02/vendor/pac-nerf'))
os.chdir(WORK)
import numpy as np, torch, mmcv
import gic_force_adapter as A
import striker_adapter as SA
from lib.engine import MPMSimulator
if os.environ.get('STEM_STRIKER_JSON'):
    print('striker adapter:', SA.install(MPMSimulator), flush=True)
else:
    print('force adapter:', A.install(MPMSimulator), flush=True)
import train as native
from lib import PACNeRF
P = json.loads((Path(__file__).parent/'protocol.json').read_text())
DX_SECTION = 0.08/6


def config(scene, outdir, n_frames, E, nu):
    cfg = mmcv.Config.fromfile(str(R/'runs/gap-crossmethod-v02/vendor/pac-nerf/configs/default.py'))['cfg']
    cfg.update(dict(dtype='float32', n_cameras=11, n_frames=n_frames, xyz_min=[-0.3, 0.45, -0.2], xyz_max=[0.3, 0.75, 0.2],
                    dx=DX_SECTION*8,  # halves three times (pg_scale) to the same final grid as GIC: section/6
                    material=MPMSimulator.elasticity, pg_scale=[1000, 2000, 4000], data_dir=str(scene), base_dir=str(outdir),
                    cuda_chunk_size=100, nerf_bs=2**16, particle_chunk_size=2**10, taichi_cuda_memory=float(os.environ.get('STEM_TI_MEM', '0.35')),
                    dt=1/48/500, frame_dt=1/48, H=800, W=800, N_static=6001, entropy_weight=1e-3, volume_weight=1e-3,
                    direct_nerf=False, E=E, nu=nu, rho=1000, hit_frame=3, physical_params=dict(global_E=1e-1, global_nu=1e-2),
                    write_out=True, random_particles=True,
                    BC={'cylinder_clamp': [[-0.30, 0.6, 0.0], [-0.20, 0.6, 0.0], 0.07]}))
    if os.environ.get('STEM_SCENE') == 'block':  # Figure-1 S2: glued block (native sticky ground plane)
        cfg.update(dict(xyz_min=[-0.2, 0.45, -0.15], xyz_max=[0.2, 0.75, 0.15], dx=0.10/8*8, E=E, nu=nu,
                        dt=1/48/300, BC={'ground': [[0, 0.53, 0], [0, 1, 0]]}))
    return cfg


def fit(scene, outdir, E=None, nu=0.1):
    E = E if E is not None else (3e4 if os.environ.get('STEM_SCENE') == 'block' else 4e5)
    scene, outdir = Path(scene), Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    n = len(json.loads((scene/'all_data.json').read_text()))//11-1
    cfg = config(scene, outdir, n, float(E), float(nu)); native.cfg = cfg
    torch.manual_seed(123); np.random.seed(123); torch.set_num_threads(4)
    t0 = time.time()
    data = native.load_data(cfg['data_dir'], 11, H=800, W=800)
    fit_data = (data[0][1:], data[1][1:], data[2][1:], data[3][1:], data[4][1:])  # camera 0 held out
    model, opt, start = native.create_model_and_optimizer(cfg, 'static', pnerf=None)
    native.train_static(cfg, model, opt, start, cfg['N_static'], *fit_data)
    model, opt, start = native.create_model_and_optimizer(cfg, 'velocity', pnerf=model)
    native.train_velocity(cfg, model, start, *fit_data)
    model, opt, start = native.create_model_and_optimizer(cfg, 'dynamic', pnerf=model)
    native.train_dynamic(cfg, model, opt, start, *fit_data)
    res = dict(E=float(10**model.global_E.item()), nu=float(native.constraint(model.global_nu, model.nu_bound).item()),
               rho=float(10**model.global_rho.item()), v=model.global_v.detach().cpu().tolist(), seconds=time.time()-t0,
               force=A._SPEC, frames=n, scene=str(scene))
    (outdir/'fit.json').write_text(json.dumps(res, indent=1)); print(res, flush=True)


def forecast(ckdir, out, nframes):
    ckdir = Path(ckdir); saved = torch.load(ckdir/'model/train_dynamic.pt', map_location='cpu', weights_only=False)
    scene = Path(saved['cfg']['data_dir']); data = torch.load(scene/'data.pt', map_location='cpu', weights_only=False)
    od = Path(out).with_suffix(''); od.mkdir(parents=True, exist_ok=True)
    cfg = config(scene, od, int(nframes), 1e5, .1); cfg.update(saved['cfg']); cfg.update(base_dir=str(od), write_out=True, nerf_bs=16384)
    model = PACNeRF(**cfg).cuda(); model.load_state_dict(saved['model_state_dict'], strict=False)
    rgb, _, _ = model.render_sequence(int(nframes), H=800, W=800, rays_o=data['rays_o'][0], rays_d=data['rays_d'][0],
                                      viewdirs=data['viewdirs'][0], savedir=str(od/'images'), dump_images=False)
    assert model.dynamic_observer.succeed() and np.isfinite(rgb).all()
    from plyfile import PlyData
    xyz = np.stack([np.stack([PlyData.read(od/'simulation'/f'{f}.ply')['vertex'][k] for k in 'xyz'], -1) for f in range(int(nframes))])
    ref = xyz[0]
    from ours import STATIONS
    cent = np.stack([[xyz[f][(ref[:, 0] >= lo) & (ref[:, 0] < hi)].mean(0) for lo, hi in STATIONS] for f in range(len(xyz))])  # stem stations (block scored from xyz)
    np.savez_compressed(out, xyz=xyz.astype(np.float32), station_centroid=cent, rgb=(np.clip(rgb, 0, 1)*255).astype(np.uint8),
                        params=json.dumps(dict(E=float(10**model.global_E.item()), nu=float(native.constraint(model.global_nu, model.nu_bound).item()),
                                               rho=float(10**model.global_rho.item()), v=model.global_v.detach().cpu().tolist())),
                        adapter_audit=json.dumps(A.AUDIT), striker=np.array(SA.STATE['history']) if SA.SPEC else np.zeros((0, 7)))
    print('forecast', out, A.AUDIT, flush=True)


if __name__ == '__main__':
    {'fit': fit, 'forecast': forecast}[sys.argv[1]](*sys.argv[2:])
