"""PAC-NeRF (Li et al. 2023, official code b613048 at runs/gap-crossmethod-v02/vendor/pac-nerf, unmodified) adapter for
PokeFlex stage B.  Run through scripts/stem_trial_v01/pac_env.sh with -I.  Reads only stage-B inputs (prep_inputs.py,
pac_prep.py): never the PokeFlex loader.

Per poke (calibration and held-out), from that poke's REST frame only:
  static   native PAC-NeRF static stage (train.train_static: hybrid Eulerian voxel NeRF <-> Lagrangian particles,
           progressive grid upsampling) on the 6 colour views of the rest frame (pac_prep.py), bbox = rest-scan bbox
           +- 15 mm; native particle generation (initialize_particles: half-voxel lattice, alpha >= 1e-3 max).
  shape    rest-shape quality: marching-cubes surface of the particle occupancy vs the rest scan (symmetric chamfer).
  physics  native MPM (lib/engine/mpm_simulator.py kernels compute_F_tmp / project_F / p2g / g2p, native elasticity
           material = compressible neo-Hookean, particle mu/lam ~ alpha^3, mass ~ alpha) with this adapter's grid
           boundary operator instead of the native static colliders: gravity, sticky support plane y <= y_s (as the
           native 'ground' BC), and the kinematic tool (rigid flat-ended cylinder r = 10 mm, axis T_WT z, end face at
           T_WT origin, pose interpolated linearly within a frame, velocity = finite difference): grid nodes inside it
           lose the velocity component moving INTO the tool relative to the tool velocity (separating contact); normal =
           tool axis for nodes within 1.5 dx above the end face, else the nearer of face/side;
           force ON THE TOOL = -(sum of the momentum removed)/dt, averaged over the last quarter of each frame.
           Density = catalogue mass / sum(alpha p_vol) (known mass, as ours).  Gravity settle 10 frames with the tool at
           its start pose; geometry reported as x_rest + (x - x_settled).
  surface  marching-cubes mesh of the rest particle occupancy, vertices moved by the kNN blend of particle
           displacements (fixed topology per poke).
Fit (calibration window only): native PAC-NeRF fits (E, nu) by gradient descent on a rendering loss; under a
displacement-driven poke the images do not constrain E (deformation is E-independent up to gravity) and PokeFlex
reconstructions do not resolve the dent, so (E, nu) are fitted by a small scan on the calibration poke with the
same loss as the PhysTwin adapter: force RMSE [N] + RMS chamfer [mm] (observed tool-excluded scan samples ->
particles).  nu in {0.0, 0.2, 0.4}; E from a force least-squares rescale (force is ~linear in E), re-simulated.
usage (via pac_env.sh -I): pfx_pacnerf.py dev | fit OBJ | predict OBJ [--shard i/n] | poke OBJ KIND TAKE K"""
import json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sb_common import (R, INP, WORK, PRED_B, R_TOOL, L_TOOL, obj_info, load_poke, save_pred, tool_pose, loading_slope)
VENDOR = R/'runs/gap-crossmethod-v02/vendor/pac-nerf'
sys.path.insert(0, str(VENDOR))
import torch  # noqa: E402
import mmcv  # noqa: E402
import taichi as ti  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from lib.engine import MPMSimulator  # noqa: E402
import train as native  # noqa: E402

METHOD = 'PAC-NeRF'
FPS = 30.0
N_STATIC = int(os.environ.get('PAC_N_STATIC', '2400'))
PG = [int(N_STATIC*f) for f in (1/6, 1/3, 2/3)]
SETTLE = 6
NUS = [0.0, 0.2, 0.4]
CHUNK = 4
MASS_SCALE = 10.0  # quasi-static pokes: inertia x10 (dt x3.2), gravity /10 so the weight stays physical
ALPHA_MIN = 0.05   # native particles have alpha >= 1e-3 max; near-transparent ones (mass ~alpha, stiffness ~alpha^3) dropped


def dx_final(V):
    vol = np.prod(V.max(0)-V.min(0))*0.55  # rough occupied volume from the scan bbox
    return float(np.clip((vol/12000)**(1/3), 0.004, 0.015))


def make_cfg(V, base_dir, E=3e4, nu=0.2):
    cfg = mmcv.Config.fromfile(str(VENDOR/'configs/default.py'))['cfg']
    lo, hi = V.min(0)-0.015, V.max(0)+0.015; lo[1] = V[:, 1].min()-0.005
    dx = dx_final(V)
    cfg.update(dict(dtype='float32', xyz_min=lo.tolist(), xyz_max=hi.tolist(), dx=dx*8, material=MPMSimulator.elasticity,
                    pg_scale=PG, base_dir=str(base_dir), data_dir=str(base_dir), cuda_chunk_size=CHUNK, nerf_bs=int(os.environ.get('PAC_NERF_BS', 2**14)),
                    particle_chunk_size=2**10, taichi_cuda_memory=float(os.environ.get('PAC_TI_MEM', '0.25')),
                    dt=1/FPS/500, frame_dt=1/FPS, H=None, W=None, N_static=N_STATIC, entropy_weight=1e-3, volume_weight=1e-3,
                    direct_nerf=False, E=E, nu=nu, rho=1000, hit_frame=3, physical_params=dict(global_E=1e-1, global_nu=1e-2),
                    write_out=False, random_particles=True, BC={}, gravity=[0, -9.81, 0]))
    return cfg


# ------------------------------------------------------------------------------------------- MPM boundary operator
@ti.kernel
def grid_op_tool(sim: ti.template(), ys: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32, ax_: ti.f32, ay: ti.f32, az: ti.f32,
                 vx: ti.f32, vy: ti.f32, vz: ti.f32, Rt: ti.f32, Lt: ti.f32, acc: ti.i32, imp: ti.template()):
    c = ti.Vector([cx, cy, cz]); a = ti.Vector([ax_, ay, az]); vt = ti.Vector([vx, vy, vz])
    for I in ti.grouped(sim.grid_m):
        if sim.grid_m[I] > 1e-8*sim.dx[None]**3:
            v = sim.grid_v_in[I]/sim.grid_m[I]+sim.dt[None]*sim.gravity[None]
            X = I.cast(ti.f32)*sim.dx[None]
            if X[1] <= ys:
                v = ti.Vector.zero(ti.f32, 3)
            else:
                u = X-c
                ax = u.dot(a)
                rad = u-ax*a
                r = rad.norm()
                if ax < 0 and ax > -Lt and r < Rt:
                    # Eulerian nodes have no history: nodes within 1.5 dx above the end face (they carry the material
                    # under the face through the B-spline support) push along the axis; deeper nodes push radially
                    # when they are closer to the side than to the face
                    n = a
                    if -ax > 1.5*sim.dx[None] and Rt-r < -ax and r > 1e-6:
                        n = rad/r
                    vn = (v-vt).dot(n)
                    if vn < 0:
                        dv = -vn*n
                        v += dv
                        if acc == 1:
                            imp[None] += ti.cast(sim.grid_m[I]*dv, ti.f64)
            sim.grid_v_out[I] = v


@ti.kernel
def roll(sim: ti.template(), last: ti.i32):
    for p in range(sim.n_particles[None]):
        sim.x[p, 0] = sim.x[p, last]; sim.v[p, 0] = sim.v[p, last]
        sim.C[p, 0] = sim.C[p, last]; sim.F[p, 0] = sim.F[p, last]


@ti.kernel
def get_x(sim: ti.template(), s: ti.i32, x: ti.types.ndarray()):
    for p in range(sim.n_particles[None]):
        for d in ti.static(range(3)):
            x[p, d] = sim.x[p, s][d]


class Scene:
    """One poke: static NeRF reconstruction of the rest frame + particles; MPM rollouts for given (E, nu)."""
    def __init__(self, obj, kind, take, k, mass):
        self.obj, self.kind, self.take, self.k = obj, kind, take, k
        src = INP/'_dev'/f'cal_{take}-p{k}.npz' if obj == '_dev' else INP/obj/f'{kind}_{take}-p{k}.npz'
        self.p = dict(np.load(src)); self.mass = mass
        self.dir = WORK/'pacnerf'/obj/f'{kind}_{take}-p{k}'
        self.cfg = make_cfg(self.p['V'], self.dir)

    def static(self):
        t0 = time.time()
        data = torch.load(self.dir/'data.pt', map_location='cpu')
        rays = (data['rays_o'], data['rays_d'], data['viewdirs'], data['rgb_all'], data['ray_mask_all'])
        S = int(round(np.sqrt(data['rays_o'].shape[1]))); self.cfg.update(H=S, W=S, n_cameras=data['rays_o'].shape[0])
        native.cfg = self.cfg
        torch.manual_seed(123); np.random.seed(123)
        model, opt, start = native.create_model_and_optimizer(self.cfg, 'static', pnerf=None)
        if start < N_STATIC:
            native.train_static(self.cfg, model, opt, start, N_STATIC, *rays)
        self.model = model
        with torch.no_grad():
            model.initialize_particles(False, data['rays_o'][0, 0].cuda())
        al = model.init_alphas.detach().float().cpu().numpy(); keep = al >= ALPHA_MIN
        self.n_particles_raw = int(len(al))
        self.alpha = al[keep]
        self.X0 = model.init_particles.detach().float().cpu().numpy()[keep]
        self.dx = float(model.nerf.voxel_size)
        self.features = model.init_features.detach()[torch.from_numpy(keep).cuda()]
        self.static_s = time.time()-t0
        self.mc = self.surface_mesh()
        return self

    def from_calibration(self, cal, model=None):
        """Held-out poke without its own static stage: the object's calibration-rest NeRF reconstruction (particles,
        alphas, surface) rigidly registered (ICP, multi-start) to this poke's rest scan (free input)."""
        t0 = time.time()
        z = np.load(cal); X, al, mcV, mcF, dx = z['X0'], z['alpha'], z['mcV'], z['mcF'], float(z['dx'])
        Rm, tv, res = icp(sample_mesh(mcV, mcF, 6000, 1), sample_mesh(self.p['V'], self.p['F'], 6000, 2))
        self.X0 = (X @ Rm.T+tv).astype(np.float32); self.alpha = al; self.dx = dx
        self.mc = self.surface_mesh()
        self.cfg.update(H=8, W=8, n_cameras=1); native.cfg = self.cfg
        self.model = model if model is not None else native.PACNeRF(**self.cfg).cuda()  # only its Taichi MPM simulator
        self.features = torch.zeros(len(X), 1+int(self.cfg['rgbnet_dim']), device='cuda')
        self.static_s = time.time()-t0; self.icp = dict(R=Rm.tolist(), t=tv.tolist(), residual_mm=res)
        return self

    def surface_mesh(self):
        """Marching cubes of the particle alpha occupancy on the half-voxel lattice (max alpha per half-cell, Gaussian
        sigma 0.6 half-cell, level 0.5), decimated by vertex clustering at the NeRF voxel size (~20k vertices; the
        decimation changes the rest chamfer by < 0.3 mm on FoamDice)."""
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter
        h = self.dx/2; lo = self.X0.min(0)-3*h
        idx = np.floor((self.X0-lo)/h).astype(int); shp = idx.max(0)+4
        occ = np.zeros(shp, np.float32); np.maximum.at(occ, tuple(idx.T), self.alpha)
        occ = gaussian_filter(occ, 0.6)
        vv, ff, _, _ = marching_cubes(occ, 0.5)
        V = lo+vv*h+0.5*h
        key = np.floor((V-V.min(0))/self.dx).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
        nv = inv.max()+1; Vc = np.zeros((nv, 3)); np.add.at(Vc, inv, V); V = Vc/np.bincount(inv, minlength=nv)[:, None]
        ff = inv[ff]; ff = ff[(ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])]
        # largest connected component (drops floaters of the real-image reconstruction)
        import scipy.sparse as sps
        from scipy.sparse.csgraph import connected_components
        e = np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]]])
        nc, lab = connected_components(sps.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
        big = np.argmax(np.bincount(lab)); fk = ff[lab[ff[:, 0]] == big]
        used = np.unique(fk); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
        self.mc_components = int(nc)
        return dict(V=V[used], F=remap[fk].astype(np.int32))

    def shape_quality(self, n=20000):
        from scipy.spatial import cKDTree as T
        rng = np.random.default_rng(0)
        def samp(V, F):
            A = 0.5*np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1)
            t = rng.choice(len(F), n, p=A/A.sum()); u, v = rng.random(n), rng.random(n); fl = u+v > 1; u[fl], v[fl] = 1-u[fl], 1-v[fl]
            return V[F[t, 0]]*(1-u-v)[:, None]+V[F[t, 1]]*u[:, None]+V[F[t, 2]]*v[:, None]
        Ps, Pt = samp(self.mc['V'], self.mc['F']), samp(self.p['V'], self.p['F'])
        a = T(Pt).query(Ps)[0].mean()*1e3; b = T(Ps).query(Pt)[0].mean()*1e3
        vol = np.prod(self.p['V'].max(0)-self.p['V'].min(0))
        return dict(nerf_to_scan_mm=float(a), scan_to_nerf_mm=float(b), chamfer_mm=float(0.5*(a+b)), n_particles=int(len(self.X0)),
                    n_particles_native=getattr(self, 'n_particles_raw', None), mc_components=getattr(self, 'mc_components', None),
                    voxel_mm=self.dx*1e3, mc_vertices=int(len(self.mc['V'])), static_seconds=self.static_s,
                    bbox_scan_m3=float(vol))

    def rollout(self, E, nu, T=None):
        """MPM with tool; returns xs (K,P,3) corrected particle positions at frames start..end, F (K,3) on the tool."""
        m = self.model; dyn = m.dynamic_observer; sim = dyn.simulator
        T = self.p['T'] if T is None else T
        tip, ax = tool_pose(T); ys = float(self.p['y_s'])
        P = len(self.X0); dev = 'cuda'
        pvol = (self.dx*0.5)**3
        rho = self.mass/(pvol*self.alpha.sum())
        mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu))
        al = torch.tensor(self.alpha, device=dev)
        sim.gravity[None] = [0.0, -9.81/MASS_SCALE, 0.0]
        dyn.initialize(torch.tensor(self.X0, device=dev), self.features, torch.zeros(P, 3, device=dev),
                       al*rho*MASS_SCALE, al**3*mu, al**3*lam, self.dx)
        cp = np.sqrt((lam+2*mu)/(rho*MASS_SCALE))
        dt = min(0.3*self.dx/cp, 1/FPS/20); n = int(np.ceil(1/FPS/dt)); dt = 1/FPS/n
        sim.dt[None] = dt
        imp = ti.Vector.field(3, ti.f64, shape=())
        frames = [(tip[0], ax[0], tip[0], ax[0])]*SETTLE+[(tip[j], ax[j], tip[j+1], ax[j+1]) for j in range(len(T)-1)]
        s = 0; xs = []; F = [np.zeros(3)]; xbuf = np.zeros((P, 3), np.float32); x_set = None
        for fi, (c0, a0, c1, a1) in enumerate(frames):
            vt = (c1-c0)*FPS; imp[None] = [0, 0, 0]; i_acc = (3*n)//4
            for i in range(n):
                w = (i+1)/n; c = (1-w)*c0+w*c1; a = (1-w)*a0+w*a1; a = a/np.linalg.norm(a)
                loc = s % CHUNK
                sim.grid.deactivate_all()
                sim.compute_F_tmp(loc); sim.project_F(loc); sim.p2g(loc)
                grid_op_tool(sim, ys, float(c[0]), float(c[1]), float(c[2]), float(a[0]), float(a[1]), float(a[2]),
                             float(vt[0]), float(vt[1]), float(vt[2]), R_TOOL, L_TOOL, int(i >= i_acc), imp)
                sim.g2p(loc)
                s += 1
                if s % CHUNK == 0:
                    roll(sim, CHUNK)
            acc = imp.to_numpy()/dt
            get_x(sim, s % CHUNK, xbuf)
            if not np.isfinite(xbuf).all():
                return None, None, dict(ok=False, nsub=n, dt=dt, frame=fi)
            if fi == SETTLE-1:
                x_set = xbuf.astype(np.float64).copy(); xs.append(self.X0.astype(np.float64).copy())
            elif fi >= SETTLE:
                xs.append(self.X0+(xbuf-x_set)); F.append(-acc/(n-(3*n)//4))
        return np.stack(xs), np.stack(F), dict(ok=True, nsub=n, dt=dt, rho=rho, settle_sag_mm=float(np.abs(x_set-self.X0)[:, 1].max()*1e3))

    def surface(self, xs, k=4):
        d, idx = cKDTree(self.X0).query(self.mc['V'], k=k)
        w = 1/np.maximum(d, 1e-6); w /= w.sum(1, keepdims=True)
        return self.mc['V'][None]+np.einsum('vk,fvkj->fvj', w, xs[:, idx]-self.X0[None, idx])


def sample_mesh(V, F, n, seed):
    rng = np.random.default_rng(seed)
    A = 0.5*np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1)
    t = rng.choice(len(F), n, p=A/A.sum()); u, v = rng.random(n), rng.random(n); fl = u+v > 1; u[fl], v[fl] = 1-u[fl], 1-v[fl]
    return V[F[t, 0]]*(1-u-v)[:, None]+V[F[t, 1]]*u[:, None]+V[F[t, 2]]*v[:, None]


def icp(src, tgt, iters=40):
    """Rigid symmetric point-to-point ICP, multi-start: 8 yaws about world y x {identity, flip about x, flip about z},
    centroids aligned.  Returns R, t, symmetric mean distance (mm) of the best start."""
    Tt = cKDTree(tgt); cs, ct = src.mean(0), tgt.mean(0); best = None
    flips = [np.eye(3), np.diag([1., -1, -1]), np.diag([-1., -1, 1])]
    for fl in flips:
        for th in np.arange(8)*np.pi/4:
            c, s_ = np.cos(th), np.sin(th); Rm = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]) @ fl; tv = ct-Rm @ cs
            for _ in range(iters):
                Y = src @ Rm.T+tv; Ts = cKDTree(Y)
                _, i1 = Tt.query(Y); _, i2 = Ts.query(tgt)
                A = np.concatenate([src, src[i2]]); B = np.concatenate([tgt[i1], tgt])
                ma, mb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A-ma).T @ (B-mb))
                D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ D @ U.T; tv = mb-Rm @ ma
            Y = src @ Rm.T+tv
            res = 0.5*(Tt.query(Y)[0].mean()+cKDTree(Y).query(tgt)[0].mean())*1e3
            if best is None or res < best[2]:
                best = (Rm, tv, float(res))
    return best


def loss_terms(p, X0, xs, F, alpha):
    sel = np.where(alpha > 0.1)[0]
    sel = sel[np.random.default_rng(0).permutation(len(sel))[:60000]]
    ch = []
    for j in range(len(xs)):
        gt = p['S'][j][p['Skeep'][j]]
        d, _ = cKDTree(xs[j][sel]).query(gt); ch.append(float((d**2).mean()))
    return float(np.mean(ch)), float(((F-p['Fmeas'])**2).sum(1).mean())


def hooke_k(p):
    a = p['approach']; d = np.maximum(p['dep'], 0)
    return float((p['Fmeas'] @ -a) @ d/(d @ d))


def fit(obj):
    info = obj_info(obj) if obj != '_dev' else dict(mass_kg=0.140, calibration_pokes=[['FoamDice_T1', 1]])
    t0 = time.time()
    t, k = info['calibration_pokes'][0]  # one calibration poke per object (others start in contact)
    sc = Scene(obj, 'cal', t, k, info['mass_kg']).static(); p = sc.p
    shape = sc.shape_quality(); print('rest shape', shape, flush=True)
    xs0 = np.repeat(sc.X0[None], len(p['frames']), 0)
    c0, f0 = loss_terms(p, sc.X0, xs0, np.zeros((len(p['frames']), 3)), sc.alpha)
    rows = []
    def run(E, nu, tag):
        t1 = time.time(); xs, F, inf = sc.rollout(E, nu)
        if xs is None:
            rows.append(dict(E=E, nu=nu, loss=float('inf'), tag=tag, **inf)); print('  unstable', E, nu, inf, flush=True); return None
        c, fe = loss_terms(p, sc.X0, xs, F, sc.alpha); L = np.sqrt(fe)+np.sqrt(c)*1e3
        cs = float((F*p['Fmeas']).sum()/max((F*F).sum(), 1e-12))
        fe_s = float(((cs*F-p['Fmeas'])**2).sum(1).mean())
        rows.append(dict(E=E, nu=nu, tag=tag, loss=L, chamfer_mse_m2=c, force_mse=fe, lsq_scale=cs, E_rescaled=E*cs,
                         loss_rescaled_est=np.sqrt(fe_s)+np.sqrt(c)*1e3, seconds=time.time()-t1, **inf))
        print(f"  [{tag}] E {E:9.1f} nu {nu:.1f} L {L:.4f} force rms {fe**.5:.2f} N chamfer rms {c**.5*1e3:.2f} mm scale {cs:.3f} "
              f"-> est L {np.sqrt(fe_s)+np.sqrt(c)*1e3:.4f} at E {E*cs:.0f}  nsub {inf['nsub']} {time.time()-t1:.0f}s", flush=True)
        return xs, F
    # one simulation per nu at a start modulus; force is ~linear in E and the deformation ~E-independent (displacement-
    # driven poke; up to gravity/inertia), so E_nu = LSQ rescale and the loss at E_nu is estimated from the rescaled
    # force; the chosen (E, nu) is then re-simulated; if an already simulated (E, nu) has a lower actual loss it is used
    E_start = hooke_k(p)/(2*R_TOOL)
    for nu in NUS:
        if run(E_start, nu, 'scan') is not None and 0.05 < rows[-1]['lsq_scale'] < 20:
            E_start = rows[-1]['E_rescaled']
    ok = [r for r in rows if np.isfinite(r['loss'])]
    best = min(ok, key=lambda r: r['loss_rescaled_est'])
    E_b, nu_b = float(best['E_rescaled']), best['nu']
    out = run(E_b, nu_b, 'final')
    if out is None:
        raise RuntimeError('final calibration rollout unstable')
    xs, F = out; best = rows[-1]
    sim_best = min((r for r in rows if np.isfinite(r['loss'])), key=lambda r: r['loss'])
    if sim_best['loss'] < best['loss']:  # the linear-in-E estimate was not realised: keep the best SIMULATED (E, nu)
        E_b, nu_b = float(sim_best['E']), sim_best['nu']
        out = run(E_b, nu_b, 'best-simulated')
        xs, F = out; best = rows[-1]
    slope = loading_slope(F @ -p['approach'], p['dep'])
    fit = dict(method=METHOD, obj=obj, calibration_pokes=[[t, k]], E=E_b, nu=nu_b, loss=best['loss'], mass_scale=MASS_SCALE, alpha_min=ALPHA_MIN,
               chamfer_null_m2=c0, force_null=f0, scan=rows, rest_shape=shape, ranking_dF_dd_N_per_m=slope,
               seconds=time.time()-t0, n_static=N_STATIC,
               loss_definition='force RMSE [N] + RMS chamfer [mm] (observed tool-excluded scan samples -> particles) on the calibration poke; 1 N <-> 1 mm')
    d = WORK/'pacnerf'/obj; (d/'fit.json').write_text(json.dumps(fit, indent=1, default=float))
    np.savez_compressed(d/f'calpred_{t}-p{k}.npz', frames=p['frames'], force=F, surf_V=sc.surface(xs).astype(np.float32), surf_F=sc.mc['F'])
    np.savez_compressed(d/'cal_nerf.npz', X0=sc.X0, alpha=sc.alpha, mcV=sc.mc['V'], mcF=sc.mc['F'], dx=sc.dx)
    if obj != '_dev':
        (PRED_B/METHOD/obj).mkdir(parents=True, exist_ok=True)
        (PRED_B/METHOD/obj/'fit.json').write_text(json.dumps(fit, indent=1, default=float))
    print(f"FIT {obj}: E {best['E']:.0f} Pa nu {best['nu']} loss {best['loss']:.4f} dF/dd {slope:.0f} N/m {fit['seconds']/60:.1f} min", flush=True)


def predict_one(obj, t, k, fit, mass, model=None):
    """Held-out poke.  PAC_HELDOUT=static: own native static stage on the poke's rest-frame images;
    PAC_HELDOUT=repose (default): calibration reconstruction registered to the poke's rest scan (see from_calibration)."""
    t0 = time.time(); mode = os.environ.get('PAC_HELDOUT', 'repose')
    sc = Scene(obj, 'ho', t, k, mass)
    if mode == 'static':
        sc.static(); shape = sc.shape_quality()
    else:
        sc.from_calibration(WORK/'pacnerf'/obj/'cal_nerf.npz', model); shape = dict(sc.shape_quality(), icp=sc.icp)
    p = sc.p
    xs, F, inf = sc.rollout(fit['E'], fit['nu'])
    sel = p['frames'] > int(p['start_frame']); flag = None
    if xs is None:
        flag = 'MPM unstable: rest NeRF surface and zero force written'
        xs = np.repeat(sc.X0[None], len(p['frames']), 0); F = np.zeros((len(p['frames']), 3))
    surf = sc.surface(xs)
    save_pred(METHOD, obj, t, k, p['frames'][sel], F[sel],
              dict(method=METHOD, heldout_geometry=mode, E=fit['E'], nu=fit['nu'], sim=inf, flag=flag, rest_shape=shape,
                   surface='marching cubes of the NeRF particle occupancy at rest, moved by kNN blend of particle displacements',
                   ranking_dF_dd_N_per_m=fit['ranking_dF_dd_N_per_m'], seconds=time.time()-t0),
              surf_V=surf[sel], surf_F=sc.mc['F'])
    print(f"  {t} p{k}: peak |F| {np.linalg.norm(F, axis=1).max():.1f} N  rest chamfer {shape['chamfer_mm']:.1f} mm  "
          f"particles {shape['n_particles']}  nsub {inf.get('nsub')}  {time.time()-t0:.0f}s", flush=True)
    return sc.model


def predict(obj, shard='0/1'):
    i, n = map(int, shard.split('/'))
    fit = json.loads((WORK/'pacnerf'/obj/'fit.json').read_text()); info = obj_info(obj); model = None
    for j, (t, k) in enumerate(info['heldout']):
        if j % n != i or (PRED_B/METHOD/obj/f'{t}-p{k}.npz').exists():
            continue
        claim = WORK/'pacnerf'/obj/f'claim_{t}-p{k}'  # several queues may work on the same object
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            continue
        model = predict_one(obj, t, k, fit, info['mass_kg'], model if os.environ.get('PAC_HELDOUT', 'repose') == 'repose' else None)
    print(f'PREDICT {obj} shard {shard} done', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dev':
        fit('_dev')
    elif cmd == 'fit':
        fit(sys.argv[2])
    elif cmd == 'predict':
        predict(sys.argv[2], *(sys.argv[3:4]))
    elif cmd == 'poke':  # one held-out poke per process (fresh Taichi runtime)
        obj, t, k = sys.argv[2], sys.argv[3], int(sys.argv[4])
        predict_one(obj, t, k, json.loads((WORK/'pacnerf'/obj/'fit.json').read_text()), obj_info(obj)['mass_kg'])
