"""Synthesise M-PhyGs physical-stage inputs (native directory layout) for one fitting experiment.
Usage: synth.py SCENE EPISODE WORLD      (EPISODE in D0, D1 only)
Writes runs/fig1-v01/mphygs/<scene>/<EP>-<world>/ with config.json and run/<exp>/... :
  physical_particle_pyramid/stage_00_grid_75/{fg_gs.ply, cluster_probabilities.pt, dino_features.pt, cluster_features.pt}
  physical_particle_tracking/stage_00_grid_75/{k:05d}.ply   (tar[k] = state after k+1 frames)
  physical_particle_contact_motion/stage_00_grid_75/{particle_modifier.pt, contact_point_00000/motion.pt}
  adapter/{inputs.npz, force_audit.json}
No gt/ directory is created (the vendor GT-leak paths stay absent)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData, PlyElement
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

STAGE = f'stage_00_grid_{C.GRID_PER_M}'


def write_ply(path, cols):
    arr = np.zeros(len(next(iter(cols.values()))), dtype=[(k, 'f4') for k in cols])
    for k, v in cols.items():
        arr[k] = v
    PlyData([PlyElement.describe(arr, 'vertex')]).write(str(path))


def exp_dir(scene, ep, world):
    return C.OUT/scene/f'{ep}-{world}'


def main(scene, ep, world):
    assert ep in C.SCENES[scene]['fit_episodes'], 'only fitting episodes may be synthesised'
    S = C.SCENES[scene]; root = exp_dir(scene, ep, world); exp = f'{scene}_{ep}_{world}'
    run = root/'run'/exp
    x, vol, sdf, pinned, body = C.sample_particles(scene)
    lab = C.kmeans_labels(x, body)
    K = lab.max()+1
    tracks, y = C.dense_tracks(scene, ep, world, x)
    F = len(tracks)
    f_full, t_ramp, audit = C.force_distribution(scene, ep, x, vol, body) if ep != 'D0' or scene == 'block' else (np.zeros_like(x), 1.0, dict(particles_loaded=0, total_force=[0, 0, 0]))
    if scene == 'stem' and ep == 'D0':
        audit['note'] = 'stem D0 is gravity release only (no applied force)'

    pp = run/'physical_particle_pyramid'/STAGE; pp.mkdir(parents=True, exist_ok=True)
    write_ply(pp/'fg_gs.ply', dict(x=x[:, 0], y=x[:, 1], z=x[:, 2], volume=vol, sdf=sdf))
    onehot = np.eye(K, dtype=np.float32)[lab]
    torch.save(torch.from_numpy(onehot), pp/'cluster_probabilities.pt')
    # "dino_features": no appearance features exist for a uniformly textured synthetic object; the native code
    # uses them only for the mat_reg neighbour graph -> we use rest coordinates (spatial neighbours).
    torch.save(torch.from_numpy(x.astype(np.float32)), pp/'dino_features.pt')
    torch.save(torch.from_numpy(onehot), pp/'cluster_features.pt')

    tr = run/'physical_particle_tracking'/STAGE; tr.mkdir(parents=True, exist_ok=True)
    for k in range(1, F):
        write_ply(tr/f'{k-1:05d}.ply', dict(x=tracks[k, :, 0], y=tracks[k, :, 1], z=tracks[k, :, 2]))

    cm = run/'physical_particle_contact_motion'/STAGE; (cm/'contact_point_00000').mkdir(parents=True, exist_ok=True)
    pin = torch.from_numpy(pinned)
    xt = torch.from_numpy(x.astype(np.float32))
    torch.save(dict(particle_modifier_mask=pin.float(), list_particle_modifier_x=xt[None].repeat(F-1, 1, 1)), cm/'particle_modifier.pt')
    torch.save(dict(init_center=xt[pin].mean(0), centers=xt[pin].mean(0)[None].repeat(F-1, 1),
                    quats=torch.tensor([[1., 0., 0., 0.]]).repeat(F-1, 1), particle_mask=pin, particle_rigid_mask=pin),
               cm/'contact_point_00000'/'motion.pt')

    ad = run/'adapter'; ad.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ad/'inputs.npz', x=x, vol=vol, sdf=sdf, pinned=pinned, body=body, labels=lab,
                        f_full=f_full, t_ramp=t_ramp, tracks=tracks, markers_obs=y)
    (ad/'force_audit.json').write_text(json.dumps(audit))

    cfg = json.loads((C.VENDOR/'confs/push_carnation.json').read_text())  # native template
    cfg.update(exp_name=exp, src_dataset_dir=str(root/'dataset'), src_gs_dataset_dir=str(root/'dataset'),
               src_gs_ply_file=str(pp/'fg_gs.ply'), num_frames=F, static_state_frame_idx=0, frame_batch_size=F-1,
               frame_timestep=1/S['fps'], sim_gravity=C.GRAVITY, sim_surface_collidar_points=[],
               sim_surface_collidar_normals=[], sim_static_part_aabbs=[], residual_gravity_sim=False, max_stiffness=C.MAX_STIFFNESS,
               adapter=dict(scene=scene, episode=ep, world=world, rho_fixed=C.RHO_FIXED, K=int(K), H=C.H))
    cfg['photometric_optimization'].update(num_frames=F, num_training_frames=F-1, train_start_frame_relative_idx=0,
                                           start_frame_idx=0)
    (root/'config.json').write_text(json.dumps(cfg, indent=1))
    assert not (root/'dataset'/'gt').exists()
    print(json.dumps(dict(exp=exp, N=len(x), body=int(body.sum()), pinned=int(pinned.sum()), K=int(K), frames=F,
                          force=audit, max_track_disp_mm=float(np.abs(tracks-x).max()*1e3))))


if __name__ == '__main__':
    main(*sys.argv[1:4])
