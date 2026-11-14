"""Thin adapter around the vendored M-PhyGs physical stage (vendor files are not modified).

Patches (all in this file, applied by subclassing / module stubs):
  * mphygs.dino_utils is replaced by a stub exposing only decode_dig_fea_img (copied verbatim): the real module
    imports transformers + nerfstudio (absent) and is only needed for DINO/DiG feature decoding of the
    photometric stage, which is skipped.
  * ForcedSimulator(PhysicsSimulator): native forward() copied verbatim except that a known per-particle body
    force is passed as particle_ext_force (native argument, added in native P2G), evaluated at the midpoint of
    each checkpoint segment with the smooth ramp 0.5(1-cos(pi t/T)). The frame index is obtained from the
    native code's own `sim_aabbs[t]` lookup via FrameList (no change to train_physical_params).
  * material_func: native one (cluster-label linear MaterialField, SDF volume scaling) with density fixed to
    1000 kg/m^3; E = (E/rho)_field * 1000. nu is native (fixed 0.3).
"""
import json, os, sys, types
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
sys.path.insert(0, str(C.VENDOR))


def _decode_dig_fea_img(fea_img, dig_nn):  # verbatim copy of mphygs.dino_utils.decode_dig_fea_img
    im = torch.stack(fea_img.unbind(-3), dim=-1)
    tar_size = list(im.size())
    y = dig_nn(im.view(-1, fea_img.size(-3)))
    tar_size[-1] = y.size(-1)
    return torch.stack(y.view(tar_size).unbind(-1), dim=-3)


_stub = types.ModuleType('mphygs.dino_utils'); _stub.decode_dig_fea_img = _decode_dig_fea_img
sys.modules['mphygs.dino_utils'] = _stub

from mphygs.sim_utils import PhysicsSimulator  # noqa: E402
from mphygs.mpm_simple.mpm_solver import MPMDifferentiableSimulationWCheckpoint  # noqa: E402
from mphygs.material_field import MaterialField  # noqa: E402
from mphygs.gs_utils import load_splatfacto_result_from_ply  # noqa: E402
import mphygs.phys_dynamic_gs as PDG  # noqa: E402

STAGE = f'stage_00_grid_{C.GRID_PER_M}'


def _stub_rasterize_gaussians(gs_params, camera_params, colors=None, **kw):
    """ADAPTER: the env's gsplat CUDA backend cannot be imported (missing `rich`, no installs allowed). The
    tracking stage renders only for photometric/feature/DT terms whose weights are 0 (photometric stage skipped),
    so rendering is replaced by a constant (zero) image of the native shape [V, H, W, C]."""
    W, Hh = np.max(np.array(camera_params['imsizes']), axis=0)
    return torch.zeros((camera_params['viewmats'].shape[0], int(Hh), int(W), colors.shape[-1]), device=colors.device)


PDG.rasterize_gaussians = _stub_rasterize_gaussians


class _NoTimer:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def disable_scoped_timers():
    """PERFORMANCE ONLY: the native solver wraps every kernel launch in wp.ScopedTimer(synchronize=True,
    print=False) purely for profiling; each forces a host-device sync, which on a GPU time-shared with other jobs
    dominates runtime. Replacing it with a no-op changes no arithmetic (kernels, order and inputs identical)."""
    import warp
    warp.ScopedTimer = _NoTimer  # mpm_solver looks it up as wp.ScopedTimer at call time


class _Clock:
    t = 0
    audit = []


CLOCK = _Clock()


class FrameList(list):
    """List of per-frame sim AABBs; the native loop indexes it with the frame number right before each
    simulator.forward call, which is how the force schedule learns the current frame."""
    def __getitem__(self, i):
        if isinstance(i, int):
            CLOCK.t = i
        return list.__getitem__(self, i)


class ForcedSimulator(PhysicsSimulator):
    def set_force(self, f_full, t_ramp, frame_dt, audit=False):
        self.f_full = None if f_full is None else torch.as_tensor(f_full, dtype=torch.float32, device=self.device)
        self.t_ramp, self.frame_dt, self.do_audit = t_ramp, frame_dt, audit

    def _ext(self, t_mid):
        if getattr(self, 'f_full', None) is None:
            return None
        a = float(C.ramp(t_mid, self.t_ramp))
        if self.do_audit:
            CLOCK.audit.append((t_mid, a, (a*self.f_full).sum(0).tolist()))
        return a*self.f_full

    # ---- native PhysicsSimulator.forward (sim_utils.py) copied; only the particle_ext_force line differs ----
    def forward(self, particle_X, particle_V, particle_F, particle_C, particle_E, particle_nu, particle_density,
                particle_rigidity=None, particle_volume=None, t0=None, dt=1/60., substep_dt=1/9600.,
                particle_modifier_mask=None, particle_modifier_x0=None, particle_modifier_x1=None,
                particle_ext_force=None, init_particle_acc=None, sim_aabb=None, aabb_diff=None,
                num_checkpoints_per_frame=1):
        assert (particle_rigidity is None) or torch.all(particle_rigidity == 0.)
        assert particle_ext_force is None
        if sim_aabb is None:
            sim_aabb = self.sim_aabb
        if aabb_diff is None:
            aabb_diff = torch.tensor([[0.0, 0.0, 0.0]], device=particle_X.device, dtype=torch.float32)
        particle_X = (particle_X - sim_aabb[0])
        sim_mask = torch.ones_like(particle_X[:, 0], dtype=torch.bool)
        if not (particle_modifier_mask is None):
            particle_modifier_x0 = particle_modifier_x0 - sim_aabb[0]
            particle_modifier_x1 = particle_modifier_x1 - sim_aabb[0]
        if not t0 is None:
            self.mpm_solver.time = t0
        if not (self.particle_static_mask is None):
            raise NotImplementedError('static_part_aabbs not used by this adapter (pinning via contact modifier)')
        grid_cell_size = self.mpm_solver.grid_length / self.mpm_solver.n_grid
        c = torch.sqrt(torch.max(self.max_stiffness * (1. - particle_nu) / (1. + particle_nu) / (1. - 2. * particle_nu))).item()
        substep_dt_ = self.substep_margin * grid_cell_size / c
        substep_dt = min(0.04999 * dt, substep_dt_)
        if not (particle_modifier_mask is None):
            particle_density = (1. - particle_modifier_mask) * particle_density + particle_modifier_mask * 1000e4
            particle_E = (1. - particle_modifier_mask) * particle_E + particle_modifier_mask * self.max_stiffness * particle_density
        num_total_substeps = int(dt / substep_dt)
        num_substeps_per_checkpoint = max(1, int((num_total_substeps + num_checkpoints_per_frame - 1) / num_checkpoints_per_frame))
        num_remaining_substeps = num_total_substeps
        self.last_substep_dt, self.last_num_substeps = substep_dt, num_total_substeps
        particle_x, particle_v, particle_f, particle_c = particle_X, particle_V, particle_F, particle_C
        t_frame0 = CLOCK.t * dt
        while num_remaining_substeps > 0:
            num_substeps_to_process = min(num_substeps_per_checkpoint, num_remaining_substeps)
            a = num_remaining_substeps / num_total_substeps
            particle_modifier_x0_ = a * particle_modifier_x0 + (1. - a) * particle_modifier_x1
            b = (num_remaining_substeps - num_substeps_to_process) / num_total_substeps
            particle_modifier_x1_ = b * particle_modifier_x0 + (1. - b) * particle_modifier_x1
            done = num_total_substeps - num_remaining_substeps
            ext = self._ext(t_frame0 + (done + 0.5 * num_substeps_to_process) * substep_dt)
            particle_x, particle_v, particle_f, particle_c = MPMDifferentiableSimulationWCheckpoint.apply(
                self.mpm_solver, substep_dt, num_substeps_to_process, particle_x, particle_v, particle_f, particle_c,
                particle_E, particle_nu, particle_density, particle_volume,
                ext, init_particle_acc,
                particle_modifier_mask, particle_modifier_x0_, particle_modifier_x1_,
                self.device, self.residual_gravity)
            num_remaining_substeps -= num_substeps_to_process
        new_particle_X = particle_x + sim_aabb[0]
        self.particle_acc_cache = (particle_v - particle_V) / dt
        return new_particle_X, particle_v, particle_f, particle_c


def load_inputs(root, exp, device):
    """Read the synthesised native-layout inputs (same files the native run() reads)."""
    base = Path(root)/'run'/exp
    pp = base/'physical_particle_pyramid'/STAGE
    p = load_splatfacto_result_from_ply(str(pp/'fg_gs.ply'), device, verbose=False)
    onehot = torch.load(pp/'cluster_probabilities.pt', weights_only=True).to(device)
    onehot = (onehot == torch.max(onehot, dim=1)[0][:, None]).float()  # native hardening
    feats = torch.load(pp/'dino_features.pt', weights_only=True).to(device)
    tr = sorted((base/'physical_particle_tracking'/STAGE).glob('*.ply'))
    tracks = [load_splatfacto_result_from_ply(str(f), device, verbose=False)['means'] for f in tr]
    cm = base/'physical_particle_contact_motion'/STAGE
    modifier = torch.load(cm/'particle_modifier.pt', weights_only=False)
    contacts = [{k: v.to(device) for k, v in torch.load(f, weights_only=False).items()}
                for f in sorted(cm.glob('contact_point_*/motion.pt'))]
    assert len(contacts) >= 1
    ad = np.load(base/'adapter'/'inputs.npz')
    return dict(x=p['means'], vol=p['volume'][:, 0], sdf=p['sdf'][:, 0], onehot=onehot, feats=feats, tracks=tracks,
                modifier_mask=modifier['particle_modifier_mask'].to(device), contacts=contacts, ad=ad)


def make_material(cfg, K, device, ckpt=None, init_E=None, rho=None):
    torch.manual_seed(32)  # native seed
    mf = MaterialField(fea_dim=K, refine_volume=cfg.get('refine_volume', False), linear=cfg['material_model'] == 'linear',
                       max_stiffness=cfg['max_stiffness'], refine_rigidity=False, refine_sdf=cfg.get('refine_sdf', False),
                       min_sdf_offset=cfg.get('min_sdf_offset', -0.005), max_sdf_offset=cfg.get('max_sdf_offset', 0.0)).to(device)
    if ckpt is not None:
        mf.load_state_dict(torch.load(ckpt)['material_field_state_dict'])
    elif init_E is not None:  # ADAPTER: same initial modulus as the PAC-NeRF/GIC runs on this scene (linear field only)
        assert isinstance(mf.linear, torch.nn.Linear)
        target = np.log(cfg['max_stiffness']/(init_E/(rho or C.RHO_FIXED)))  # = softplus(a)
        with torch.no_grad():
            mf.linear.weight[0].zero_(); mf.linear.bias[0] = float(target+np.log(-np.expm1(-target)))  # softplus^-1
    return mf


def make_material_func(mf, inp, rho=C.RHO_FIXED):
    def material_func():
        E, nu, dens, vsf, rig, sdf_off = mf({'cluster_label': inp['onehot']}, debug=False)
        stiff = E / dens  # native normalized stiffness E/rho
        dens = torch.full_like(dens, rho)  # ADAPTER: density fixed (GIC/PAC protocol)
        vol = inp['vol'] * vsf
        vol = vol * (1e-4 + torch.sigmoid(-5e3 * (inp['sdf'] + sdf_off)))
        return {'particle_E': stiff * rho, 'particle_nu': nu, 'particle_density': dens, 'particle_volume': vol,
                'particle_rigidity': rig}
    return material_func


def truth_material_func(inp, E_field, nu=0.3, rho=C.RHO_FIXED):
    E = torch.as_tensor(E_field, dtype=torch.float32, device=inp['x'].device)
    def material_func():
        vol = inp['vol'] * (1e-4 + torch.sigmoid(-5e3 * inp['sdf']))
        return {'particle_E': E, 'particle_nu': torch.full_like(E, nu), 'particle_density': torch.full_like(E, rho),
                'particle_volume': vol, 'particle_rigidity': torch.zeros_like(E)}
    return material_func


def sim_aabbs_from(frames):
    """Native rule (alternating_photo_opt.py): per-frame bbox centres, size = 2 x max extent."""
    X = torch.stack(frames, 0)
    bb = torch.stack([X.min(1)[0], X.max(1)[0]], 1)
    cen = bb.mean(1); size = 2. * torch.max(bb[:, 1] - bb[:, 0])
    aabbs = torch.stack([cen - .5 * size, cen + .5 * size], 1)
    return aabbs, (cen - cen[0:1])[:, None], size


def make_simulator(cfg, x0, aabb0, size, device):
    sim = ForcedSimulator(x0, sim_aabb=aabb0, device=device, grid_size=int(C.GRID_PER_M * size.item()),
                          grid_length=size.item(), surface_collidar_points=[], surface_collidar_normals=[],
                          static_part_aabbs=[], gravity=cfg['sim_gravity'], residual_gravity=False,
                          max_stiffness=cfg['max_stiffness'])
    return sim


def dummy_photometric(inp, T, device, stride=16, im=32):
    """Minimal synthetic photometric inputs so the native tracking loop runs; with photo/fea/dt weights 0 these
    contribute exactly 0 to the loss and gradient (checked in smoke test)."""
    xg = inp['x'][::stride].contiguous(); n = len(xg)
    gs = dict(means=xg, quats=torch.tensor([[1., 0, 0, 0]], device=device).repeat(n, 1),
              scales=torch.full((n, 3), float(np.log(0.004)), device=device),
              opacities=torch.full((n, 1), 2.0, device=device), rgb_colors=torch.full((n, 3), 0.5, device=device))
    mask = torch.ones(n, dtype=torch.bool, device=device)
    feats = torch.zeros((n, 4), device=device)  # 3 rgb + 4 feat + 1 mask = 8 channels
    c = xg.mean(0)
    Rm = torch.diag(torch.tensor([1., -1., -1.], device=device))
    vm = torch.eye(4, device=device); vm[:3, :3] = Rm; vm[:3, 3] = -Rm @ (c + torch.tensor([0, 0, 1.], device=device))
    Ks = torch.tensor([[im, 0, im/2], [0, im, im/2], [0, 0, 1]], device=device, dtype=torch.float32)
    cam = dict(viewmats=vm[None], Ks=Ks[None], imsizes=[[im, im]])
    ref = [[torch.full((3, im, im), 0.5, device=device) for _ in range(T)]]
    fea = [[torch.zeros((4, im//4, im//4), device=device) for _ in range(T)]]
    dts = [[torch.zeros((im, im), device=device) for _ in range(T)]]
    return dict(gs=gs, mask=mask, feats=feats, cams=[cam]*T, ref=ref, fea=fea, dts=dts)


def cluster_params(mf_func, inp):
    with torch.no_grad():
        mv = mf_func()
    lab = inp['onehot'].argmax(1).cpu().numpy()
    E = mv['particle_E'].cpu().numpy(); d = mv['particle_density'].cpu().numpy(); nu = mv['particle_nu'].cpu().numpy()
    pinned = inp['modifier_mask'].cpu().numpy() > 0
    out = {}
    for k in np.unique(lab[~pinned]):
        s = (lab == k) & ~pinned
        out[int(k)] = dict(E=float(E[s][0]), density=float(d[s][0]), nu=float(nu[s][0]), n=int(s.sum()),
                           centroid=inp['x'][torch.from_numpy(s).to(inp['x'].device)].mean(0).tolist())
    return out


disable_scoped_timers()  # always on (performance only, see docstring)
