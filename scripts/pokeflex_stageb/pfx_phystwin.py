"""PhysTwin (Jiang et al. 2025) adapter for PokeFlex stage B.  Env: runs/phystwin-env; kernels from the official
upstream copy runs/third_party/PhysTwin-v06 (unmodified): eval_springs, update_vel_from_force,
integrate_ground_collision.  Reads only stage-B inputs (prep_inputs.py, ptw_geom.py): never the PokeFlex loader.

Model (native PhysTwin spring-mass, see stageB-adapters.md):
  nodes      surface + interior nodes of the poke's rest scan (ptw_geom.py), native UNIT mass per node; physical node
             mass m = catalogue mass / N, so a physical force f enters as f/m and physical spring stiffness = Y_nat * m.
  springs    native topology rule: every node connects to <= max_neighbours nodes within radius (hybrid search),
             unique pairs, rest length = rest-scan distance.  Global stiffness parameterised physically (Y_phys, N),
             so it transfers between pokes (h fixed per object).
  frame      native frame: ground plane z=0, gravity -z (gravity_scale 1, reverse_z False); world -> native
             (x, y - y_s, z) -> (x, -z, y - y_s), y_s = lowest rest-scan vertex (support plane, as ours).
  support    bonded, as ours: nodes within 0.75 h of the support plane held at rest (rigid version of native fixed
             controller points); every other node keeps the native ground collision (collide_elas, collide_fric fitted).
             (A first run with ground collision only let objects slide/tip under the tool: 3dPrintedPyramid calibration
             peak force 5 N vs 64 N measured; it was replaced for all objects.)
  tool       kinematic rigid flat-ended cylinder r = 10 mm, axis T_WT z, end face at the T_WT origin (as ours); penalty
             on nodes inside it, pushed out through the end face or radially (side the node came from, as ours'
             K_SWITCH rule); tool pose interpolated linearly within a frame.  Force ON THE TOOL = -(sum of penalty
             forces), averaged over the last quarter of each frame's substeps, world frame.
  gravity    settle phase (0.5 s, tool at its start pose) before the poke; reported geometry = rest scan + (x - x_settled)
             (the scan is the loaded shape; same convention as ours).
Fit (native stage A = CMA-ES over global parameters, sigma 1/6, normalised box): log Y_phys, radius factor,
max_neighbours, collide_elas, collide_fric, dashpot, drag.  Loss = force RMSE (N, 3-vector, calibration frames) +
RMS of the native chamfer distance (mm; observed calibration-frame scan samples, tool-excluded, to the predicted surface
nodes): the protocol's two scored units, exchanged 1 N <-> 1 mm.  Native stage B (per-spring Adam) is not run: each held-out poke builds its own springs
from its own rest scan, so only global parameters transfer.
usage: python -I pfx_phystwin.py dev | fit OBJ | predict OBJ | all OBJ"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sb_common import (R, INP, WORK, PRED_B, R_TOOL, L_TOOL, obj_info, load_poke, save_pred, tool_pose, loading_slope)
os.environ['PHYSTWIN_HEADLESS'] = '1'; os.environ['WANDB_MODE'] = 'disabled'
UP = R/'runs/third_party/PhysTwin-v06'
sys.path.insert(0, str(UP))
import warp as wp  # noqa: E402
import qqtt.model.diff_simulator.spring_mass_warp as K  # noqa: E402  native kernels

METHOD = 'PhysTwin'
FPS = 30.0
K_SWITCH = 100.0
SETTLE_FRAMES = 15
RN = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])  # world -> native rotation
DEFAULTS = dict(r_fac=2.0, max_nn=30, collide_elas=0.5, collide_fric=0.3, dashpot=100.0, drag=3.0)
BOX = dict(logY=None, r_fac=(1.2, 3.0), max_nn=(10, 50), collide_elas=(0.0, 1.0), collide_fric=(0.0, 2.0),
           dashpot=(0.0, 300.0), drag=(0.0, 10.0))
NAMES = ['logY', 'r_fac', 'max_nn', 'collide_elas', 'collide_fric', 'dashpot', 'drag']
wp.init()
DEV = 'cuda:0'


@wp.kernel
def tool_penalty(x: wp.array(dtype=wp.vec3), tips: wp.array(dtype=wp.vec3), axes: wp.array(dtype=wp.vec3),
                 step: wp.array(dtype=wp.int32), R: float, L: float, ksw: float, kpen: float, inv_m: float,
                 nsub: int, avg_from: int, side: wp.array(dtype=wp.int32), f: wp.array(dtype=wp.vec3),
                 fsum: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    s = step[0]
    c = tips[s]
    a = axes[s]
    u = x[i]-c
    ax = wp.dot(u, a)
    rad = u-ax*a
    r = wp.length(rad)
    if ax < 0.0 and ax > -L and r < R:
        dface = -ax
        dside = R-r
        wf = dface
        ws = dside
        if side[i] == 1:
            ws = ksw*dside
        else:
            wf = ksw*dface
        n = a
        d = dface
        if ws < wf:
            n = rad/wp.max(r, 1e-9)
            d = dside
        fp = kpen*d*n
        f[i] = f[i]+fp*inv_m
        if s % nsub >= avg_from:
            wp.atomic_add(fsum, 0, wp.vec3d(wp.float64(fp[0]), wp.float64(fp[1]), wp.float64(fp[2])))
    else:
        if ax >= 0.0:
            side[i] = 1
        else:
            side[i] = 0


@wp.kernel
def pin(x0: wp.array(dtype=wp.vec3), fixed: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if fixed[i] == 1:
        x[i] = x0[i]
        v[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def inc(step: wp.array(dtype=wp.int32)):
    step[0] = step[0]+1


def springs_for(X, radius, max_nn):
    """Native PhysTwin topology (trainer_warp._init_start): hybrid radius / max-neighbour search, unique pairs."""
    d, idx = cKDTree(X).query(X, k=int(max_nn)+1, distance_upper_bound=radius)
    i = np.repeat(np.arange(len(X)), idx.shape[1]-1); j = idx[:, 1:].ravel(); dd = d[:, 1:].ravel()
    ok = np.isfinite(dd) & (dd > 1e-4)
    i, j = i[ok], j[ok]
    pairs = np.unique(np.sort(np.stack([i, j], 1), 1), axis=0)
    rest = np.linalg.norm(X[pairs[:, 0]]-X[pairs[:, 1]], axis=1)
    return pairs.astype(np.int32), rest.astype(np.float32)


class Sim:
    def __init__(self, geom, mass, prm):
        self.g = geom; self.prm = prm
        Xw = geom['X']; self.y_s = float(geom['y_s']); self.N = len(Xw)
        self.Xn = self.to_n(Xw)
        # bonded support (as ours): nodes within 0.75 h of the support plane are held at rest (= native fixed controller
        # points with rigid coupling); all other nodes keep the native ground collision
        self.fixed = self.Xn[:, 2] < 0.75*float(geom["h"])
        self.m = mass/self.N
        self.springs, self.rest = springs_for(Xw, prm['r_fac']*float(geom['h']), prm['max_nn'])
        self.Ynat = 10**prm['logY']/self.m
        ksum = np.bincount(self.springs.ravel(), weights=np.repeat(self.Ynat/self.rest, 2), minlength=self.N)
        omega = float(np.sqrt(ksum.max()))
        nnb = np.bincount(self.springs.ravel(), minlength=self.N).max()
        dt = min(5e-5, 0.25/omega, 0.5/max(prm['dashpot']*nnb, 1e-9))
        n = int(np.ceil(1/FPS/dt)); n += n % 2
        self.nsub = n; self.dt = 1/FPS/n
        self.kpen = 4*self.m*omega**2  # penalty stiffness per node, stable at dt*2*omega = 0.5
        self.omega = omega

    def to_n(self, Xw):
        return (Xw-np.array([0, self.y_s, 0])) @ RN.T

    def to_w(self, Xn):
        return Xn @ RN+np.array([0, self.y_s, 0])

    def run(self, T, settle=SETTLE_FRAMES):
        """T (K,4,4) tool poses at frames start..end. Returns xs (K,N,3) world nodes (rest + x - x_settled), F (K,3)."""
        tip, ax = tool_pose(T); tip = self.to_n(tip); ax = ax @ RN.T
        n = self.nsub; Kf = len(T)
        s = (np.arange(1, n+1)/n)[:, None]
        tips = [np.repeat(tip[:1], settle*n, 0)]; axs = [np.repeat(ax[:1], settle*n, 0)]
        for j in range(Kf-1):
            tips.append((1-s)*tip[j]+s*tip[j+1]); a = (1-s)*ax[j]+s*ax[j+1]; axs.append(a/np.linalg.norm(a, axis=1, keepdims=True))
        tips = wp.array(np.concatenate(tips).astype(np.float32), dtype=wp.vec3, device=DEV)
        axs = wp.array(np.concatenate(axs).astype(np.float32), dtype=wp.vec3, device=DEV)
        N = self.N
        xA = wp.array(self.Xn.astype(np.float32), dtype=wp.vec3, device=DEV); vA = wp.zeros(N, dtype=wp.vec3, device=DEV)
        xrest = wp.array(self.Xn.astype(np.float32), dtype=wp.vec3, device=DEV)
        fixed = wp.array(self.fixed.astype(np.int32), dtype=wp.int32, device=DEV)
        xB = wp.zeros_like(xA); vB = wp.zeros_like(vA); vt = wp.zeros_like(vA); f = wp.zeros_like(vA)
        side = wp.zeros(N, dtype=wp.int32, device=DEV); step = wp.zeros(1, dtype=wp.int32, device=DEV)
        fsum = wp.zeros(1, dtype=wp.vec3d, device=DEV)
        masses = wp.array(np.ones(N, np.float32), dtype=wp.float32, device=DEV)
        springs = wp.array(self.springs, dtype=wp.vec2i, device=DEV); rest = wp.array(self.rest, dtype=wp.float32, device=DEV)
        logY = wp.array(np.full(len(self.rest), np.log(self.Ynat), np.float32), dtype=wp.float32, device=DEV)
        elas = wp.array([self.prm['collide_elas']], dtype=wp.float32, device=DEV); fric = wp.array([self.prm['collide_fric']], dtype=wp.float32, device=DEV)
        dummy = wp.zeros(1, dtype=wp.vec3, device=DEV)
        avg_from = (3*n)//4; dt = float(self.dt)
        def substep(xa, va, xb, vb):
            f.zero_()
            wp.launch(tool_penalty, dim=N, inputs=[xa, tips, axs, step, R_TOOL, L_TOOL, K_SWITCH, float(self.kpen), float(1/self.m),
                                                    n, avg_from, side, f, fsum], device=DEV)
            wp.launch(K.eval_springs, dim=len(self.rest), inputs=[xa, va, dummy, dummy, N, springs, rest, logY,
                      float(self.prm['dashpot']), 0.0, 1e30, f], device=DEV)
            wp.launch(K.update_vel_from_force, dim=N, inputs=[va, f, masses, dt, float(self.prm['drag']), 1.0, vt], device=DEV)
            wp.launch(K.integrate_ground_collision, dim=N, inputs=[xa, vt, elas, fric, dt, 1.0, xb, vb], device=DEV)
            wp.launch(pin, dim=N, inputs=[xrest, fixed, xb, vb], device=DEV)
            wp.launch(inc, dim=1, inputs=[step], device=DEV)
        with wp.ScopedCapture(device=DEV) as cap:
            for i in range(n//2):
                substep(xA, vA, xB, vB); substep(xB, vB, xA, vA)
        for _ in range(settle):
            wp.capture_launch(cap.graph)
        x0 = xA.numpy().astype(np.float64)
        xs = [self.Xn.copy()]; F = [np.zeros(3)]
        for j in range(Kf-1):
            fsum.zero_(); wp.capture_launch(cap.graph)
            x = xA.numpy().astype(np.float64)
            xs.append(self.Xn+(x-x0)); F.append(-fsum.numpy()[0]/(n-avg_from))
        xs = np.stack(xs)
        ok = bool(np.isfinite(xs).all())
        Fw = np.stack(F) @ RN  # native -> world
        return self.to_w(xs), Fw, dict(ok=ok, n_fixed=int(self.fixed.sum()), nsub=n, dt=dt, kpen=float(self.kpen), n_springs=int(len(self.rest)),
                                       settle_sag_mm=float(np.abs(x0-self.Xn)[:, 2].max()*1e3))


# ------------------------------------------------------------------------------------------------ loss / fit
def surface(geom, xs):
    """Rest-scan vertices moved by the kNN blend of node displacements (K,Nv,3)."""
    V0 = geom['V']; idx, w = geom['interp_idx'], geom['interp_w']
    D = xs-geom['X'][None]
    return V0[None]+np.einsum('vk,fvkj->fvj', w, D[:, idx])


def loss_terms(cal, geom, xs, F):
    ns = int(geom['n_surface']); ch = []
    for j in range(len(xs)):
        gt = cal['S'][j][cal['Skeep'][j]]
        d, _ = cKDTree(xs[j, :ns]).query(gt); ch.append(float((d**2).mean()))
    fe = float(((F-cal['Fmeas'])**2).sum(1).mean())
    return float(np.mean(ch)), fe


def params_from(x, logY_range):
    x = np.clip(x, 0, 1); p = {}
    for i, k in enumerate(NAMES):
        lo, hi = logY_range if k == 'logY' else BOX[k]
        p[k] = lo+x[i]*(hi-lo)
    p['max_nn'] = int(round(p['max_nn']))
    return p


def to_x(p, logY_range):
    return np.array([(p[k]-(logY_range if k == 'logY' else BOX[k])[0])/((logY_range if k == 'logY' else BOX[k])[1]-(logY_range if k == 'logY' else BOX[k])[0]) for k in NAMES])


def load_geom(obj, kind, t, k):
    g = dict(np.load(WORK/'phystwin'/obj/f'geom_{kind}_{t}-p{k}.npz'))
    p = load_poke('_dev' if obj == '_dev' else obj, kind, t, k) if obj != '_dev' else dict(np.load(INP/'_dev'/f'cal_{t}-p{k}.npz'))
    g['V'] = p['V']; g['F'] = p['F']
    return g, p


class Objective:
    def __init__(self, obj, mass, pokes):
        self.obj, self.mass = obj, mass
        self.data = [load_geom(obj, 'cal', t, k) for t, k in pokes]
        self.null = []
        for g, p in self.data:
            xs0 = np.repeat(g['X'][None], len(p['frames']), 0)
            self.null.append(loss_terms(p, g, xs0, np.zeros((len(p['frames']), 3))))
        self.evals = []

    def __call__(self, prm, keep=False):
        t0 = time.time(); L = 0.0; terms = []; outs = []
        for (g, p), (c0, f0) in zip(self.data, self.null):
            sim = Sim(g, self.mass, prm)
            xs, F, info = sim.run(p['T'])
            if not info['ok']:
                L = float('inf'); terms.append(None); break
            c, fe = loss_terms(p, g, xs, F)
            L += np.sqrt(fe)+np.sqrt(c)*1e3; terms.append(dict(chamfer_mse_m2=c, chamfer_null=c0, force_mse=fe, force_null=f0, **info))
            if keep:
                outs.append((xs, F))
        L /= len(self.data)
        self.evals.append(dict(prm=prm, loss=L, terms=terms, seconds=time.time()-t0))
        print(f"  eval {len(self.evals):3d} L {L:8.4f} logY {prm['logY']:.2f} r {prm['r_fac']:.2f} nn {prm['max_nn']} "
              f"el {prm['collide_elas']:.2f} fr {prm['collide_fric']:.2f} dp {prm['dashpot']:.0f} dr {prm['drag']:.1f} "
              f"| {terms[0] and round(terms[0]['force_mse']**.5, 2)} N rms, {terms[0] and round(terms[0]['chamfer_mse_m2']**.5*1e3, 2)} mm rms "
              f"| nsub {terms[0] and terms[0]['nsub']} {time.time()-t0:.1f}s", flush=True)
        return (L, outs) if keep else L


def hooke_k(p):
    a = p['approach']; d = np.maximum(p['dep'], 0)
    return float((p['Fmeas'] @ -a) @ d/(d @ d))


def fit(obj, cma_gens=12, scan=None):
    import cma
    info = obj_info(obj) if obj != '_dev' else dict(mass_kg=0.140, calibration_pokes=[['FoamDice_T1', 1]])
    pokes = info['calibration_pokes']
    t0 = time.time(); ob = Objective(obj, info['mass_kg'], pokes)
    g, p = ob.data[0]
    Y_guess = hooke_k(p)*float(g['h'])
    # coarse scan of the global stiffness at native defaults: 0.5-decade grid from 3 decades below the Hooke-slope
    # guess Y ~ k h up to the guess (soft to stiff; stops once the loss is > 3x the best and rising, stiff runs are
    # expensive), then +-0.25 decade around the best; the CMA box is +-0.5 decade around the best value
    l0 = float(np.log10(Y_guess)); rows = []
    for off in np.arange(-3.0, 0.01, 0.5):
        v = ob(dict(DEFAULTS, logY=l0+off)); rows.append((v, l0+off))
        if len(rows) > 2 and v > 3*min(r[0] for r in rows) and v > rows[-2][0]:
            break
    bl = min(rows)[1]
    for off in (-0.25, 0.25):
        rows.append((ob(dict(DEFAULTS, logY=bl+off)), bl+off))
    best_l = min(rows)[1]
    rng = (best_l-0.5, best_l+0.5)
    x0 = to_x(dict(DEFAULTS, logY=best_l), rng)
    es = cma.CMAEvolutionStrategy(list(np.clip(x0, 0, 1)), 1/6, dict(bounds=[0, 1], seed=42, maxiter=cma_gens, verbose=-9))
    while not es.stop():
        X = es.ask(); es.tell(X, [ob(params_from(np.array(x), rng)) for x in X])
    cands = [e for e in ob.evals if np.isfinite(e['loss'])]
    best = min(cands, key=lambda e: e['loss'])
    prm = best['prm']
    L, outs = ob(prm, keep=True)
    slopes = []; cal_out = []
    for (g, p), (xs, F) in zip(ob.data, outs):
        slopes.append(loading_slope(F @ -p['approach'], p['dep']))
        cal_out.append(dict(xs=xs, F=F))
    fit = dict(method=METHOD, obj=obj, calibration_pokes=pokes, params=prm, loss=L, terms=ob.evals[-1]['terms'],
               prescan=[dict(logY=l, loss=v) for v, l in rows], logY_box=rng, cma_generations=cma_gens, n_evals=len(ob.evals),
               ranking_dF_dd_N_per_m=float(np.mean(slopes)), seconds=time.time()-t0,
               loss_definition='mean over calibration pokes of force RMSE [N] + RMS chamfer [mm] (observed tool-excluded scan samples -> predicted surface nodes); 1 N <-> 1 mm',
               history=[dict(loss=e['loss'], prm=e['prm'], seconds=e['seconds']) for e in ob.evals])
    d = WORK/'phystwin'/obj; d.mkdir(parents=True, exist_ok=True)
    (d/'fit.json').write_text(json.dumps(fit, indent=1, default=float))
    for (t, k), (g, p), co in zip(pokes, ob.data, cal_out):
        np.savez_compressed(d/f'calpred_{t}-p{k}.npz', frames=p['frames'], force=co['F'], surf_V=surface(g, co['xs']).astype(np.float32),
                            surf_F=g['F'])
    if obj != '_dev':
        (PRED_B/METHOD/obj).mkdir(parents=True, exist_ok=True)
        (PRED_B/METHOD/obj/'fit.json').write_text(json.dumps({k: v for k, v in fit.items() if k != 'history'}, indent=1, default=float))
    print(f'FIT {obj}: loss {L:.4f} params {prm} dF/dd {fit["ranking_dF_dd_N_per_m"]:.0f} N/m  {fit["seconds"]/60:.1f} min', flush=True)
    return fit


def predict(obj):
    info = obj_info(obj); fit = json.loads((WORK/'phystwin'/obj/'fit.json').read_text()); prm = fit['params']
    t0 = time.time(); n = 0
    for t, k in info['heldout']:
        g, p = load_geom(obj, 'ho', t, k)
        sim = Sim(g, info['mass_kg'], prm); xs, F, inf = sim.run(p['T'])
        sel = p['frames'] > int(p['start_frame'])
        surf = surface(g, xs)
        flag = None
        if not inf['ok']:
            flag = 'non-finite rollout: rest surface and zero force written'
            surf = np.repeat(p['V'][None], len(xs), 0); F = np.zeros_like(F)
        save_pred(METHOD, obj, t, k, p['frames'][sel], F[sel],
                  dict(method=METHOD, params=prm, sim=inf, flag=flag, surface='rest scan moved by kNN blend of node displacements',
                       ranking_dF_dd_N_per_m=fit['ranking_dF_dd_N_per_m']), surf_V=surf[sel], surf_F=p['F'])
        n += 1
        print(f'  {t} p{k}: peak |F| {np.linalg.norm(F, axis=1).max():.1f} N  nsub {inf["nsub"]} ok {inf["ok"]}', flush=True)
    print(f'PREDICT {obj}: {n} held-out pokes, {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dev':
        fit('_dev', cma_gens=int(sys.argv[2]) if len(sys.argv) > 2 else 2, scan=int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    elif cmd == 'fit':
        fit(sys.argv[2])
    elif cmd == 'predict':
        predict(sys.argv[2])
    elif cmd == 'all':
        fit(sys.argv[2]); predict(sys.argv[2])
