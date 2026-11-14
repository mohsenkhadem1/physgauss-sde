"""Figure-1 held-out-camera (camera 0) rendering scores, reported beside the physics scores.

Protocol line: "camera 0 (never used for fitting) foreground PSNR, reported beside physics; appearance per
method's own native renderer". Only SEALED predictions are rendered (fig1_score.verify / sealed lookup).

Subcommands
  refs                     (.venv)    truth camera-0 references, written as camera-0-only PAC-NeRF scenes
                                      runs/fig1-v01/render-cam0/scenes/{EP}-{W} (same renderer/texture/camera as
                                      every fitting scene; no striker sphere: the existing pipeline never renders it)
  gic DIR FIT LABEL MODE SCEN W [SCEN W ...]   (GIC env, cwd = GIC vendor dir)
                                      MODE shared: one fixed appearance (GIC's fixed-particle Gaussians, trained by
                                      GIC's own stage on cameras 1-10 of its fitting episode) driven by every
                                      method's sealed motion -> isolates motion.
                                      MODE native: only the GIC forecast that lives on this model's own particles
                                      (GIC's native renderer for that forecast) plus true motion (its own ceiling).
  raster SCEN              (.venv)    secondary: the true rest mesh + procedural texture (privileged appearance,
                                      the reference renderer itself) driven by each method's motion; ceiling = exact.
  score                    (.venv)    -> reports/fig1-v01/render-scores.json

Motion transfer (every method, same rule): displacement of each target point (Gaussian rest centre, or truth mesh
node for raster) = inverse-distance-squared weighted mean of the displacements (xyz[f]-xyz[0], or the FE field U)
of the K=4 nearest source rest points (method's frame-0 particles / FE nodes). Exact identity when the target
points ARE the source particles (GIC forecast through its own model). True motion is given both through the same
IDW rule and through exact T10 FE interpolation at rest points clamped into the true body (round-2 rule)."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fig1_score as FS

ST, BK, EXT = FS.ST, FS.BK, FS.EXT
OUT = R/'runs/fig1-v01/render-cam0'; REP = R/'reports/fig1-v01'
SCEN = {'S1': ('D2c', FS.SW), 'S2': ('D2', FS.BW), 'S3': ('D3', FS.SW)}
K_NN = 4
STAGE = {'D0': 'before calibration (D0 only)', 'D1': 'after calibration (D0+D1)', 'oracle': 'oracle (true parameters)'}


def sources(scen, w):
    """Sealed motion predictions for one scenario/world: (method, stage, path, kind, key).
    kind 'particles' -> npz['xyz'] (F,N,3), rest = xyz[0]; kind 'field' -> npz['X'] + npz[key] (F,n,3)."""
    ep = SCEN[scen][0]; s = []
    if scen in ('S1', 'S3'):
        s += [('GIC', 'D0', ST/f'gic/forecast-{ep}-D0.npz', 'particles', None),
              ('GIC', 'D1', ST/f'gic/forecast-{ep}-D1-{w}.npz', 'particles', None),
              ('GIC simulator, true parameters', 'oracle', ST/f'gic/forecast-{ep}-oracle-{w}.npz', 'particles', None),
              ('PAC-NeRF', 'D0', ST/f'pac/forecast-{ep}-D0.npz', 'particles', None),
              ('PAC-NeRF', 'D1', ST/f'pac/forecast-{ep}-D1-{w}.npz', 'particles', None)]
    else:
        for st in ('D0', 'D1'):
            s += [('GIC', st, BK/f'gic/forecast-D2-{st}-{w}.npz', 'particles', None),
                  ('PAC-NeRF', st, BK/f'pac/forecast-D2-{st}-{w}.npz', 'particles', None)]
    for e in FS.EXTERNAL:
        for st in ('D0', 'D1'):
            s.append((FS.EXT_NAME[e], st, EXT/e/f'forecast-{ep}-{st}-{w}.npz', 'particles', None))
    if scen == 'S1':
        s += [('ours (RGB)', 'D0', ST/f'ours-fields/round3-rgb-{w}-D2c.npz', 'field', 'D0'),
              ('ours (RGB)', 'D1', ST/f'ours-fields/round3-rgb-{w}-D2c.npz', 'field', 'D0pD1')]
    elif scen == 'S2':
        s += [('ours (RGB)', 'D0', BK/f'ours/field-rgb-{w}.npz', 'field', 'D0'),
              ('ours (RGB)', 'D1', BK/f'ours/field-rgb-{w}.npz', 'field', 'D0pD1')]
    else:
        s += [('ours (RGB)', 'D0', ST/f'ours/striker/field-rgb-{w}-D0.npz', 'field', 'U'),
              ('ours (RGB)', 'D1', ST/f'ours/striker/field-rgb-{w}-D0pD1.npz', 'field', 'U')]
    return s


def sealed_sources(scen, w, sealed):
    ok, skipped = [], []
    for src in sources(scen, w):
        (ok if sealed(src[2]) else skipped).append(src)
    return ok, skipped


def load_motion(src):
    _, _, p, kind, key = src; z = np.load(p)
    if kind == 'particles':
        x = z['xyz'].astype(np.float64); return x[0], x-x[0]
    return z['X'].astype(np.float64), z[key].astype(np.float64)


def truth_path(scen, w):
    ep = SCEN[scen][0]
    return (BK if scen == 'S2' else ST)/f'truth/{ep}-{w}.npz'


def truth_mesh(scen, z):
    import fem
    if scen == 'S2':
        import block as B; return B.mesh(tuple(int(c) for c in z['cells']))
    return fem.Mesh(tuple(int(c) for c in z['cells']))


def idw(P, D, A, k=K_NN):
    """Displacements D (F,N,3) at rest points P (N,3) -> (F,M,3) at targets A (M,3)."""
    from scipy.spatial import cKDTree
    d, i = cKDTree(P).query(A, k=k)
    wgt = 1/np.maximum(d, 1e-6)**2; wgt /= wgt.sum(1, keepdims=True)
    return np.einsum('mk,fmkj->fmj', wgt, D[:, i])


def fe_interp(scen, z, A):
    m = truth_mesh(scen, z); eps = 1e-6
    I = m.interpolation(np.clip(A, m.lo+eps, m.hi-eps))
    U = z['U']; return np.stack([I@U[f] for f in range(len(U))])


# ---------------------------------------------------------------- truth references
def refs():
    import render
    cams, K = render.cameras()
    for scen, (ep, worlds) in SCEN.items():
        for w in worlds:
            out = OUT/f'scenes/{ep}-{w}'
            if (out/'all_data.json').exists():
                continue
            z = np.load(truth_path(scen, w)); m = truth_mesh(scen, z); assert np.allclose(m.X, z['X'])
            fps = 48
            render.write_scene(out, z['X'][None]+z['U'], z['X'], m.surface_triangles(), fps, cams=[cams[0]], K=K)
            (out/'render-meta.json').write_text(json.dumps(dict(scenario=scen, episode=ep, world=w, camera=0, frames=int(len(z['U'])),
                                                               truth_sha256=FS.sha(truth_path(scen, w)), striker_sphere_rendered=False)))
            print('ref', out, flush=True)


# ---------------------------------------------------------------- GIC appearance renders (GIC env)
def gic(args):
    model, fit, label, mode = Path(args[0]), Path(args[1]), args[2], args[3]; jobs = list(zip(args[4::2], args[5::2]))
    seals = {s: FS.verify(s) for s in {j[0] for j in jobs}}  # hashes checked before anything is rendered
    import gic_stem_render as GR  # installs the camera-0 hold-out partition into GIC's PAC-NeRF reader
    from argparse import Namespace, ArgumentParser
    import torch
    from gaussian_renderer import render
    from train_gs_fixed_pcd import train_gs_with_fixed_pcd
    from new_trajectory import load_pcd_file
    from arguments import OptimizationParams
    cfg = eval((model/'cfg_args').read_text(), {'Namespace': Namespace}); cfg.model_path = str(model)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([0., 0, 0], device='cuda')  # GIC trains on black-composited images (white_background False)
    assert not cfg.white_background
    # GIC's reader trains on data/m_*.png. PAC-NeRF preprocessing overwrote m_ in place (matted) in the shared
    # scene folders, so appearance TRAINING must see m_ == r_ (the original renders); block models use clean copies.
    # (Stem appearances were trained 2026-11-09 ~10:30, before PAC-NeRF touched those scenes; they are only loaded.)
    from PIL import Image
    for r in (sorted((Path(cfg.source_path)/'data').glob('r_*.png')) if not (model/'stem-fixpcd-trained.json').exists() else []):
        assert np.array_equal(np.asarray(Image.open(r)), np.asarray(Image.open(str(r).replace('/r_', '/m_')))), f'matted m_ image: {r}'
    phys = json.loads((model/'0-pred.json').read_text())
    vol = load_pcd_file(str(model), 40000)
    parser = ArgumentParser(); op = OptimizationParams(parser); opt = op.extract(parser.parse_args(['--reg_scale', '--reg_alpha']))
    xyz = torch.tensor(np.load(fit)['xyz'], device='cuda')
    assert torch.allclose(xyz[0], vol, atol=1e-6), 'fit trajectory must start at the reconstructed particles'
    marker = model/'stem-fixpcd-trained.json'; t0 = time.time()
    scene = train_gs_with_fixed_pcd(vol, cfg, opt, pipe, [], [opt.iterations], [x-vol for x in xyz], phys['fps'],
                                    force_train=not marker.exists(), grid_size=phys['density_grid_size'])
    if not marker.exists():
        marker.write_text(json.dumps(dict(iterations=opt.iterations, fit_traj=str(fit), frames=len(xyz), train_seconds=time.time()-t0)))
    print(label, 'appearance ready', round(time.time()-t0), 's', flush=True)
    A = vol.cpu().numpy().astype(np.float64)
    out = OUT/'renders/gic'; out.mkdir(parents=True, exist_ok=True)
    for scen, w in jobs:
        ok, _ = sealed_sources(scen, w, seals[scen]); ep = SCEN[scen][0]
        z = np.load(truth_path(scen, w)); m = truth_mesh(scen, z)
        src = {('true motion', 'fe-interp'): fe_interp(scen, z, A), ('true motion', 'idw'): idw(m.X, z['U'], A)}
        for s in ok:
            P, D = load_motion(s)
            same = P.shape == A.shape and np.allclose(P, A, atol=1e-6)
            if mode == 'native' and not (s[0] == 'GIC' and same):
                continue
            src[(f'{s[0]}|{s[1]}', 'identity' if same else 'idw')] = D if same else idw(P, D, A)
        views = GR.cam0(OUT/f'scenes/{ep}-{w}', cfg)
        with torch.no_grad():
            for (lab, tr), d in src.items():
                f_ = out/f'{label}|{scen}|{w}|{lab}|{tr}.npz'
                if f_.exists():
                    continue
                d = torch.tensor(d, device='cuda', dtype=torch.float32); assert len(d) == len(views), (lab, len(d), len(views))
                imgs = []
                for f, view in enumerate(views):
                    view.load2device()
                    imgs.append(GR.to_u8(render(view, scene.gaussians, pipe, bg, d[f], 0., 0., False)['render'])); view.load2device('cpu')
                np.savez_compressed(f_, images=np.stack(imgs))
        print(label, scen, w, [k[0] for k in src], flush=True)


# ---------------------------------------------------------------- true-texture rasteriser (secondary)
def raster(scen):
    import render
    sealed = FS.verify(scen); cams, K = render.cameras(); ep, worlds = SCEN[scen]
    out = OUT/'renders/raster'; out.mkdir(parents=True, exist_ok=True)
    for w in worlds:
        z = np.load(truth_path(scen, w)); m = truth_mesh(scen, z); tris = m.surface_triangles(); X = z['X']
        ok, _ = sealed_sources(scen, w, sealed)
        for s in ok:
            f_ = out/f'{scen}|{w}|{s[0]}|{s[1]}|idw.npz'
            if f_.exists():
                continue
            P, D = load_motion(s); U = idw(P, D, X)
            imgs = [np.round(render.render(cams[0], K, X+U[f], X, tris)[0]*255).astype(np.uint8) for f in range(len(U))]
            np.savez_compressed(f_, images=np.stack(imgs)); print('raster', scen, w, s[0], s[1], flush=True)


# ---------------------------------------------------------------- scoring
def ref_images(ep, w):
    from PIL import Image
    d = OUT/f'scenes/{ep}-{w}/data'; n = len(list(d.glob('r_0_*.png')))-1
    return np.stack([np.asarray(Image.open(d/f'r_0_{f}.png')) for f in range(n)])  # uint8, white background


def psnr(pred, ref_u8, black):
    """Round-2 metric. Foreground mask = reference (truth) camera-0 pixels that are not pure white (GIC reader rule).
    MSE pooled over frames 1..end (frame 0 is the shared, given rest state) on that mask, and full frame.
    black=True: reference composited onto black (GIC renders on black, as it is trained); False: white (PAC, raster)."""
    mask = ref_u8.astype(int).sum(-1) != 765
    ref = ref_u8/255*(mask[..., None] if black else 1)
    e = (pred.astype(float)/255-ref)**2
    per = [float(-10*np.log10(e[f][mask[f]].mean())) for f in range(1, len(e))]
    return dict(fg_psnr_db=float(-10*np.log10(np.mean([e[f][mask[f]].mean() for f in range(1, len(e))]))),
                full_psnr_db=float(-10*np.log10(e[1:].mean())), fg_psnr_per_frame=per,
                fg_pixels_mean=float(mask[1:].sum((1, 2)).mean()))


def score():
    rows, skipped = [], []
    seals = {s: FS.verify(s) for s in SCEN}
    load = lambda p: np.load(p)['images']
    for scen, (ep, worlds) in SCEN.items():
        for w in worlds:
            ref = ref_images(ep, w); ok, sk = sealed_sources(scen, w, seals[scen])
            skipped += [dict(scenario=scen, world=w, method=s[0], stage=s[1], file=str(s[2].relative_to(R)), reason='not in sealed manifest') for s in sk]
            # GIC appearance renders (shared + GIC native)
            for f in sorted((OUT/'renders/gic').glob(f'*|{scen}|{w}|*.npz')):
                parts = f.stem.split('|'); app = parts[0]; method = parts[3]; stage = parts[4] if len(parts) == 6 else 'truth'; tr = parts[-1]
                shared = app.startswith('shared')
                rows.append(dict(scenario=scen, episode=ep, world=w, method=method, stage=stage,
                                 renderer='shared GIC appearance' if shared else 'GIC native (own fixed-particle Gaussians)',
                                 appearance=app, transfer=tr, **psnr(load(f), ref, black=True)))
            # PAC-NeRF native renders (camera 0 of its fitting scene; identical camera in every scene)
            for s in ok:
                if s[0] == 'PAC-NeRF':
                    rgb = np.load(s[2])['rgb']; assert rgb.shape == ref.shape, (rgb.shape, ref.shape)
                    rows.append(dict(scenario=scen, episode=ep, world=w, method='PAC-NeRF', stage=s[1], renderer='PAC-NeRF native (render_sequence, camera 0)',
                                     appearance='PAC-NeRF own fit', transfer='identity', **psnr(rgb, ref, black=False)))
            for f in sorted((OUT/'renders/raster').glob(f'{scen}|{w}|*.npz')):
                _, _, method, stage, tr = f.stem.split('|')
                rows.append(dict(scenario=scen, episode=ep, world=w, method=method, stage=stage, renderer='true-texture rasteriser (privileged appearance)',
                                 appearance='true rest mesh + texture', transfer=tr, **psnr(load(f), ref, black=False)))
    summary = {}
    for r in rows:
        k = (r['scenario'], r['renderer'], r['appearance'].split('-')[0] if r['renderer'].startswith('shared') else r['renderer'], r['method'], r['stage'], r['transfer'])
        summary.setdefault(k, []).append(r)
    phys = {}  # physics RMS beside rendering (read-only from fig1-scores.json)
    for p in json.loads((REP/'fig1-scores.json').read_text())['rows']:
        st = {'passive only': 'D0', 'after poke': 'D1', 'oracle': 'oracle'}[p['evidence']]
        phys[(p['scenario'], p['world'], p['method'], st)] = p['rms_mm']
    for r in rows:
        if (r['scenario'], r['world'], r['method'], r['stage']) in phys:
            r['physics_rms_mm'] = phys[(r['scenario'], r['world'], r['method'], r['stage'])]
    summ = []
    for k, v in summary.items():
        d = dict(scenario=k[0], renderer=k[1], method=k[3], stage=k[4], transfer=k[5], worlds=[r['world'] for r in v],
                 fg_psnr_db_mean=float(np.mean([r['fg_psnr_db'] for r in v])), full_psnr_db_mean=float(np.mean([r['full_psnr_db'] for r in v])),
                 fg_psnr_db_by_world={r['world']: r['fg_psnr_db'] for r in v})
        if all('physics_rms_mm' in r for r in v):
            d['physics_rms_mm_mean'] = float(np.mean([r['physics_rms_mm'] for r in v]))
        summ.append(d)
    not_rendered = [dict(scenario='S2', method='fitted spatially varying FEM (control)',
                         reason='sealed output is 27 top-surface probe displacements only (pred-fem-*.npz); no full field or particles to drive an appearance; re-simulating after truth opened would not be a sealed prediction'),
                    dict(scenario='all', method='M-PhyGs, PhysTwin, ours (RGB)', reason='native renderer: no native renderer in adapter (shared-appearance and rasteriser rows only)')]
    REP.joinpath('render-scores.json').write_text(json.dumps(dict(
        protocol=json.loads((HERE/'protocol_fig1.json').read_text())['rendering'],
        freezes={s: json.loads((REP/f'prediction-freeze-{s}.json').read_text())['sealed_utc'] for s in SCEN},
        metric='foreground PSNR (dB) on camera 0; mask = truth-render pixels not pure white; MSE pooled over frames 1..end; full-frame PSNR also given',
        transfer=f'IDW (1/d^2, K={K_NN} nearest source rest points); identity when the model renders its own particles; true motion also via exact FE interpolation (fe-interp)',
        summary_over_worlds=summ, rows=[{k: v for k, v in r.items() if k != 'fg_psnr_per_frame'} for r in rows],
        per_frame={f"{r['scenario']}|{r['world']}|{r['renderer']}|{r['appearance']}|{r['method']}|{r['stage']}|{r['transfer']}": r['fg_psnr_per_frame'] for r in rows},
        skipped_unsealed=skipped, not_rendered=not_rendered), indent=1))
    for s in summ:
        print(f"{s['scenario']} {s['renderer'][:28]:28s} {s['method'][:30]:30s} {s['stage']:7s} {s['transfer']:9s} fg {s['fg_psnr_db_mean']:6.2f} full {s['full_psnr_db_mean']:6.2f} {len(s['worlds'])}")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'refs':
        refs()
    elif cmd == 'gic':
        gic(sys.argv[2:])
    elif cmd == 'raster':
        raster(sys.argv[2])
    else:
        score()
