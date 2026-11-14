"""M-PhyGs tracking-stage material fit (native train_physical_params, --opt_settings mat, single chunk).
Usage: fit.py SCENE EPISODE WORLD --frames N [--iters K] [--init CKPT] [--suffix S]
Mirrors one call of `python -m scripts.alternating_photo_opt --skip_photometric --num_outer_iters 1
--opt_settings mat --num_training_frames N --frame_batch_size N --grid_size 75 --uniform_gravity
[--material_checkpoint_path CKPT]`; output dir follows the native naming."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mphygs_lib as L
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene'); ap.add_argument('episode'); ap.add_argument('world')
    ap.add_argument('--frames', type=int, required=True, help='num_training_frames (simulated frames after rest)')
    ap.add_argument('--iters', type=int, default=None, help='override num_iters of the mat setting (smoke tests)')
    ap.add_argument('--init', default=None, help='material_checkpoint_path')
    ap.add_argument('--checkpoints-per-frame', type=int, default=10)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--init-E', type=float, default=None, help='initial modulus (Pa) when no checkpoint; matches PAC-NeRF/GIC init')
    ap.add_argument('--pos-ref-m', type=float, default=0.01, help='rescale pos_loss_weight by (ref/rms observed displacement)^2; 0 = native weight')
    a = ap.parse_args()
    assert a.episode in ('D0', 'D1')
    dev = 'cuda:0'
    root = C.OUT/a.scene/f'{a.episode}-{a.world}'; exp = f'{a.scene}_{a.episode}_{a.world}'
    cfg = json.loads((root/'config.json').read_text())
    assert not Path(cfg['src_dataset_dir'], 'gt').exists(), 'GT-leak path present'
    inp = L.load_inputs(root, exp, dev)
    ad = inp['ad']
    T = a.frames; assert T <= len(inp['tracks'])
    tracks = inp['tracks'][:T]
    out_dir = root/'run'/exp/f'alternating_photo_opt_{T}frames_bs{T}_grid{C.GRID_PER_M}_uniform_gravity{a.suffix}'
    curr = out_dir/'tracking_00_mat'/'chunk_00000'
    curr.mkdir(parents=True, exist_ok=True)

    K = inp['onehot'].shape[1]
    mf = L.make_material(cfg, K, dev, a.init, init_E=a.init_E)
    mfunc = L.make_material_func(mf, inp)
    aabbs, diffs, size = L.sim_aabbs_from(tracks)
    sim = L.make_simulator(cfg, inp['x'], aabbs[0], size, dev)
    frame_dt = cfg['frame_timestep']
    sim.set_force(ad['f_full'], float(ad['t_ramp']), frame_dt)
    opts = dict(dt=frame_dt, substep_dt=cfg['sim_timestep'], num_checkpoints_per_frame=a.checkpoints_per_frame)
    stage = dict(cfg['alternating_photo_opt']['settings_per_stage'][0]); assert stage['stage_name'] == 'tracking'
    setting = dict(**stage, **cfg['alternating_photo_opt']['opt_settings']['mat'])
    if a.iters is not None:
        setting['num_iters'] = a.iters
    # Scale-free position loss: native weights were tuned on cm-scale flower motion; our block moves ~1 mm, which
    # left the regularisers dominant and the fit frozen. Same rule for every scene, from the observed tracks only.
    rms_obs = float(torch.sqrt(((torch.stack(tracks, 0)-inp['x'][None])**2).sum(-1).mean()))
    pos_scale = (a.pos_ref_m/rms_obs)**2 if a.pos_ref_m > 0 else 1.0
    setting['pos_loss_weight'] = setting['pos_loss_weight']*pos_scale
    ph = L.dummy_photometric(inp, T, dev)
    x0 = inp['x']
    init_states = dict(particle_x=x0, particle_v=torch.zeros_like(x0),
                       particle_F=torch.eye(3, device=dev)[None].repeat(len(x0), 1, 1),
                       particle_C=torch.zeros((len(x0), 3, 3), device=dev))
    mat_space = dict(particle_x=x0, particle_v=torch.zeros_like(x0), particle_F_quats1=torch.zeros((len(x0), 4), device=dev),
                     particle_F_quats2=torch.zeros((len(x0), 4), device=dev), particle_F_scales=torch.zeros((len(x0), 3), device=dev),
                     particle_C=torch.zeros((len(x0), 3, 3), device=dev))
    mat_space = L.PDG.decode_particle_params(mat_space)
    contacts = [{k: (v[:T] if k in ('centers', 'quats') else v) for k, v in c.items()} for c in inp['contacts']]
    sdf = inp['sdf']
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    L.PDG.train_physical_params(
        ph['gs'], ph['gs'], ph['mask'], ph['feats'], mat_space, init_states, torch.stack(tracks, 0),
        inp['modifier_mask'], list(mf.parameters()), mfunc, sim, opts, ph['ref'], ph['fea'], ph['dts'], ph['cams'],
        str(curr), lambda z: z,
        num_iter=setting['num_iters'], num_knn_particles=cfg.get('num_knn_particles', 8),
        material_lr=setting['material_lr'], init_particle_x_lr=setting['init_particle_x_lr'],
        init_particle_v_lr=setting['init_particle_v_lr'], init_particle_F_lr=setting['init_particle_F_lr'],
        init_particle_C_lr=setting['init_particle_C_lr'], init_particle_acc_lr=setting['init_particle_acc_lr'],
        contact_pos_lr=setting['contact_pos_lr'], contact_quat_lr=setting['contact_quat_lr'],
        pos_loss_weight=setting['pos_loss_weight'], pos_loss_mode=setting.get('pos_loss_mode', 'point_wise'),
        photo_loss_weight=setting['photo_loss_weight'], fea_loss_weight=setting['fea_loss_weight'],
        mask_loss_weight=setting.get('mask_loss_weight', 0.),
        chunk_consistency_x_loss_weight=setting['chunk_consistency_x_loss_weight'],
        chunk_consistency_v_loss_weight=setting['chunk_consistency_v_loss_weight'],
        chunk_consistency_F_loss_weight=setting['chunk_consistency_F_loss_weight'],
        chunk_consistency_C_loss_weight=setting['chunk_consistency_C_loss_weight'],
        chunk_consistency_acc_loss_weight=setting['chunk_consistency_acc_loss_weight'],
        stiffness_var_loss_weight=setting['stiffness_var_loss_weight'],
        density_var_loss_weight=setting['density_var_loss_weight'], mat_reg_loss_weight=setting['mat_reg_loss_weight'],
        rigidity_correlation_loss_weight=setting.get('rigidity_correlation_loss_weight', 0.),
        mat_reg_neighbor_weight_coef=setting['mat_reg_neighbor_weight_coef'], particle_features=inp['feats'],
        contact_pos_smoothness_loss_weight=setting['contact_pos_smoothness_loss_weight'],
        contact_quat_smoothness_loss_weight=setting['contact_quat_smoothness_loss_weight'],
        num_photometric_loss_scales=100, loss_grad_smoothing_beta=0.999, loss_grad_threshold=1e-6,
        idx_chunk=0, num_chunks=1, list_contact_motion_data=contacts, sim_aabbs=L.FrameList(list(aabbs)),
        sim_aabb_diffs=list(diffs), init_particle_acc=torch.zeros_like(x0), reference_masks=None,
        reference_human_mask_frames=None, mask_human=False,
        particle_sdf=sdf if torch.any(sdf > 0) else None)
    torch.save({'material_field_state_dict': mf.state_dict()}, curr/'material_field.ckpt')
    lt = json.loads((curr/'losses_transition.json').read_text())
    params = dict(scene=a.scene, episode=a.episode, world=a.world, frames=T, iters=setting['num_iters'],
                  init=a.init, init_E=a.init_E, rms_obs_m=rms_obs, pos_loss_scale=pos_scale, clusters=L.cluster_params(mfunc, inp), seconds=time.time()-t0,
                  peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9, substep_dt=sim.last_substep_dt,
                  substeps_per_frame=sim.last_num_substeps,
                  loss_pos_first=lt['pos'][0], loss_pos_last=lt['pos'][-1],
                  photo_fea_dts_terms=[max(lt[k]) for k in ('photo', 'fea', 'dts')],
                  photo_grad_dp_max_abs=float(np.max(np.abs(lt['photo_grad_dp']))))
    (curr/'params.json').write_text(json.dumps(params, indent=1))
    print(json.dumps({k: v for k, v in params.items() if k != 'clusters'}))
    print({k: round(v['E']) for k, v in params['clusters'].items()})


if __name__ == '__main__':
    main()
