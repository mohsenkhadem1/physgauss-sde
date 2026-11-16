"""'ours' and 'fem_lsq' on PokeFlex (protocol amendment 2026-11-14, methods_v01), built on realfem.py.

  sim     simulate every calibration-window poke and every held-out poke of every object at E_REF for each nu in NUS
          (force scales with E, deformation is E-independent up to gravity -> one run per (poke, nu)).  Free inputs
          only (rest scan of the poke, robot trajectory, catalogue mass).  Workers claim jobs through O_EXCL claim
          files, so any number of units can run the same command.  Low nu first (cheaper; full grid still required).
  fit     per object, from the calibration window only: posterior over (log E grid, nu) with plug-in noise;
          fem_lsq point estimate.  -> pred/{ours,fem_lsq}/<Obj>/fit.json
  predict held-out predictions (force, surface, risk) -> pred/{ours,fem_lsq}/<Obj>/<take>-p<k>.{npz,json}
usage: python -I pfx_ours.py sim | fit | predict | status"""
import json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pokeflex as P
import realfem as RF
from pfx_common import OBJECTS, PRED, approach_axis, heldout_pokes, calibration_pokes, calibration_force, contact_depth, poke, save_pred

SIMS = P.OUT/'sims'
NUS = [0.0, 0.1, 0.2, 0.3, 0.4]
SNAP_MIN_FRAMES = 2       # a simulation with >= this many snap frames is treated as failed for that (poke, nu)
BUDGET_S_PER_FRAME = 60.0  # realfem hard time limit per frame, at least 2400 s (dev take: median ~9, max ~67 s per frame)


def sim_file(obj, take, k, nu):
    return SIMS/obj/f'{take}-p{k}-nu{nu:.1f}.npz'


def jobs():
    out = []
    for nu in NUS:
        for obj in OBJECTS:
            for t, k in calibration_pokes(obj)+heldout_pokes(obj):
                if contact_depth(obj, t, k)[1][0] > 0:  # starts in contact: dented rest scan; not usable (fit excludes it)
                    continue
                out.append((obj, t, k, nu))
    return out


def run_sims():
    """PFX_RETRY_BUDGET_S=<s>: second pass over first-pass SimTimeout failures only, with budget <s> (compute-only
    change; .failed is renamed .timeout1 on success, a second failure is written to .failed2)."""
    host = f'{os.uname().nodename}-{os.getpid()}'
    retry = float(os.environ.get('PFX_RETRY_BUDGET_S', 0))
    J = jobs()
    if retry:  # priority: calibration pokes, then pokes with the fewest successful nu (none = no prediction at all)
        cal = {(o, t, k) for o in OBJECTS for t, k in calibration_pokes(o)}
        nok = {(o, t, k): sum(sim_file(o, t, k, n).exists() for n in NUS) for o, t, k, _ in J}
        J = sorted(J, key=lambda j: ((j[0], j[1], j[2]) not in cal, nok[j[:3]], j[3]))
    for obj, t, k, nu in J:
        f = sim_file(obj, t, k, nu)
        if f.exists():
            continue
        if retry:
            if not f.with_suffix('.failed').exists() or f.with_suffix('.failed2').exists() \
                    or 'SimTimeout' not in f.with_suffix('.failed').read_text():
                continue
        elif f.with_suffix('.failed').exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        claim = f.with_suffix('.claim')
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, host.encode()); os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            m = RF.build(obj, t, k)
            q = poke(obj, t, k); n = q['end_idx']-q['start_idx']+1
            r = RF.simulate(m, obj, t, k, nu, budget_s=retry or max(2400.0, BUDGET_S_PER_FRAME*n))
            tmp = f.with_name(f.stem+f'.tmp{os.getpid()}.npz')
            np.savez_compressed(tmp, **{a: (v.astype(np.float32) if a == 'surf_V' else v) for a, v in r.items() if a != 'log'},
                                log=np.array(r['log'], dtype=str), h=m.h, cells=len(m.T)//6)
            tmp.replace(f)
            if retry:
                f.with_suffix('.failed').replace(f.with_suffix('.timeout1'))
            print(f'{obj} {t} p{k} nu{nu}: {r["seconds"]:.0f}s frames {len(r["frames"])} conv {int(r["converged"].sum())} '
                  f'degraded {int((r["degraded"] > 0).sum())} snap {int(r["snap"].sum())} peak {np.abs(r["force"]).max():.1f}N', flush=True)
        except Exception as e:  # recorded, not hidden; the job stays unclaimed-failed for inspection
            f.with_suffix('.failed2' if retry else '.failed').write_text(f'{type(e).__name__}: {e}\n')
            print(f'{obj} {t} p{k} nu{nu}: FAILED {type(e).__name__}: {e} after {time.time()-t0:.0f}s', flush=True)
        finally:
            claim.unlink(missing_ok=True)


def status():
    J = jobs(); done = [j for j in J if sim_file(*j).exists()]
    print(f'{len(done)}/{len(J)} simulations done')
    for nu in NUS:
        d = [j for j in done if j[3] == nu]; print(f'  nu {nu}: {len(d)}/{len(J)//len(NUS)}')
    fails = list(SIMS.glob('*/*.failed')); claims = list(SIMS.glob('*/*.claim'))
    print(f'  running {len(claims)}  failed {len(fails)}')
    sec = [float(np.load(sim_file(*j))['seconds']) for j in done]
    if sec:
        print(f'  seconds per sim: median {np.median(sec):.0f}  max {max(sec):.0f}  total {sum(sec)/3600:.1f} h')


# ---------------------------------------------------------------- fit (calibration window only)
E_GRID = np.logspace(2, 7, 401)
SURF_EVERY = 2  # every 2nd calibration frame for the surface term
N_RISK = 2000


def load_sim(obj, t, k, nu):
    f = sim_file(obj, t, k, nu)
    if not f.exists():
        return None
    z = dict(np.load(f))
    tr = P.robot_trajectory(obj, t)['T_WT']; T = tr[np.asarray(z['frames'])-1]; ax = T[:, :3, 2]/np.linalg.norm(T[:, :3, 2], axis=1)[:, None]
    z['snap'] = RF.snap_frames(np.einsum('ij,ij->i', z['force'], -ax), T[:, :3, 3], ax)  # current detector, not the stored flag
    z['snapped'] = int(z['snap'].sum()) >= SNAP_MIN_FRAMES
    return z


def along(obj, t, F):
    return F @ -approach_axis(obj, t)


def fit_object(obj, nus=NUS):
    cal, off, n_air = calibration_force(obj)
    started_in_contact = [[t, k] for t, k, fr, Fm, dep in cal if dep[0] > 0]
    cal = [c for c in cal if c[4][0] <= 0]
    assert cal, f'{obj}: every calibration poke starts in contact'
    rows = {}
    for nu in nus:
        sims = [load_sim(obj, t, k, nu) for t, k, *_ in cal]
        if any(z is None for z in sims):
            rows[nu] = dict(status='missing'); continue
        if any(z['snapped'] for z in sims):
            rows[nu] = dict(status='snapped on a calibration poke (excluded)'); continue
        ff = fF = FF = 0.0; n = 0; slope_num = slope_den = 0.0; surf = []
        for (t, k, fr, Fm, dep), z in zip(cal, sims):
            assert np.array_equal(np.asarray(z['frames']), fr), (obj, t, k)
            f = z['force']; ff += float((f*f).sum()); fF += float((f*Fm).sum()); FF += float((Fm*Fm).sum()); n += f.size
            d = np.maximum(dep, 0); fa = along(obj, t, f); slope_num += float(fa@d); slope_den += float(d@d)
            Tr = P.robot_trajectory(obj, t)['T_WT']
            for j in range(0, len(fr), SURF_EVERY):
                Vt, Ft, conf = P.mesh(obj, t, int(fr[j])); Tk = Tr[fr[j]-1]
                e = P.surface_error_mm(z['surf_V'][j].astype(float), z['surf_F'], Vt, Ft, conf_t=conf, tool_tip=Tk[:3, 3], tool_axis=Tk[:3, 2])
                surf.append(e['chamfer_excl_tool'])
        c = E_GRID/RF.E_REF
        sse = c**2*ff-2*c*fF+FF  # sum over frames and xyz of |c f - F|^2
        rows[nu] = dict(status='ok', sse_F=sse, n_F=n, sse_S=float(np.sum(np.square(surf))), n_S=len(surf),
                        surf_mean_mm=float(np.mean(surf)), slope_ref=slope_num/slope_den if slope_den else np.nan,
                        E_lsq=float(RF.E_REF*fF/ff), rmse_lsq=float(np.sqrt((FF-fF**2/ff)/(n/3))))
    ok = [nu for nu in nus if rows[nu]['status'] == 'ok']
    assert ok, f'{obj}: no usable nu'
    s2F = min(rows[nu]['sse_F'].min()/rows[nu]['n_F'] for nu in ok)       # plug-in noise (per component)
    s2S = min(rows[nu]['sse_S']/rows[nu]['n_S'] for nu in ok)             # plug-in noise (surface, mm^2)
    logp = np.full((len(nus), len(E_GRID)), -np.inf)
    for i, nu in enumerate(nus):
        if nu in ok:
            logp[i] = -rows[nu]['sse_F']/(2*s2F)-rows[nu]['sse_S']/(2*s2S)
    post = np.exp(logp-logp.max()); post /= post.sum()
    w_nu = post.sum(1); Emean_nu = np.where(w_nu > 0, (post*E_GRID).sum(1)/np.maximum(w_nu, 1e-300), np.nan)
    i_ls = int(np.argmin([rows[nu]['rmse_lsq'] if nu in ok else np.inf for nu in nus]))
    stiff = float(sum(w_nu[i]*Emean_nu[i]/RF.E_REF*rows[nu]['slope_ref'] for i, nu in enumerate(nus) if nu in ok))
    fit = dict(calibration_pokes=[[t, k] for t, k, *_ in cal], excluded_started_in_contact=started_in_contact, offset_N=off.tolist(), offset_frames=n_air,
               noise_force_N=float(np.sqrt(s2F)), noise_surface_mm=float(np.sqrt(s2S)),
               posterior_nu={f'{nu:.1f}': float(w_nu[i]) for i, nu in enumerate(nus)},
               posterior_mean_E_given_nu={f'{nu:.1f}': (float(Emean_nu[i]) if np.isfinite(Emean_nu[i]) else None) for i, nu in enumerate(nus)},
               posterior_mean_E=float((post*E_GRID).sum()), posterior_mean_logE_sd=float(np.sqrt((post*(np.log10(E_GRID)-(post*np.log10(E_GRID)).sum())**2).sum())),
               ranking_dF_dd_N_per_m=stiff,
               per_nu={f'{nu:.1f}': {a: v for a, v in rows[nu].items() if a != 'sse_F'} for nu in nus},
               fem_lsq=dict(nu=nus[i_ls], E=rows[nus[i_ls]]['E_lsq'], calibration_force_rmse_N=rows[nus[i_ls]]['rmse_lsq']),
               E_grid=[float(E_GRID[0]), float(E_GRID[-1]), len(E_GRID)])
    (PRED.parent/'posterior').mkdir(exist_ok=True)
    np.savez(PRED.parent/'posterior'/f'{obj}.npz', post=post, E=E_GRID, nu=np.array(nus))
    for meth in ('ours', 'fem_lsq'):
        (PRED/meth/obj).mkdir(parents=True, exist_ok=True); (PRED/meth/obj/'fit.json').write_text(json.dumps(fit, indent=1))
    return fit, post


def run_fit():
    for obj in OBJECTS:
        fit, _ = fit_object(obj)
        print(f"{obj:18s} E {fit['posterior_mean_E']/1e3:8.2f} kPa (sd log10 {fit['posterior_mean_logE_sd']:.3f})  nu "
              f"{ {k: round(v, 2) for k, v in fit['posterior_nu'].items()} }  noise {fit['noise_force_N']*np.sqrt(3):.2f} N  "
              f"lsq nu {fit['fem_lsq']['nu']} E {fit['fem_lsq']['E']/1e3:.2f} kPa rmse {fit['fem_lsq']['calibration_force_rmse_N']:.2f} N  "
              f"dF/dd {fit['ranking_dF_dd_N_per_m']:.0f} N/m", flush=True)


# ---------------------------------------------------------------- held-out predictions
def run_predict():
    rng = np.random.default_rng(0)
    for obj in OBJECTS:
        z = np.load(PRED.parent/'posterior'/f'{obj}.npz'); post, E = z['post'], z['E']
        fit = json.loads((PRED/'ours'/obj/'fit.json').read_text()); s2F = fit['noise_force_N']**2
        lsq = fit['fem_lsq']; n_fb = 0; missing = []
        for t, k in heldout_pokes(obj):
            sims = {nu: load_sim(obj, t, k, nu) for nu in NUS}
            valid = [i for i, nu in enumerate(NUS) if sims[nu] is not None and not sims[nu]['snapped'] and post[i].sum() > 0]
            flag = None
            if not valid:
                valid = [i for i, nu in enumerate(NUS) if sims[nu] is not None and post[i].sum() > 0]; flag = 'every nu snapped; snapped runs used'
            if valid:
                p = post[valid]; p = p/p.sum()
            else:  # no simulated nu of this poke carries posterior mass (its calibration sims failed at those nu):
                # E-marginal posterior applied uniformly to the nu that this poke does have (non-snapped preferred)
                valid = [i for i, nu in enumerate(NUS) if sims[nu] is not None and not sims[nu]['snapped']] or \
                        [i for i, nu in enumerate(NUS) if sims[nu] is not None]
                if not valid:
                    missing.append(f'{t}-p{k}'); print(f'{obj} {t} p{k}: no simulation at any nu -> no prediction', flush=True); continue
                pE = post.sum(0); p = np.outer(np.full(len(valid), 1/len(valid)), pE/pE.sum())
                flag = 'no simulated nu in posterior support: E-marginal posterior applied uniformly over the available nu'
            w = p.sum(1); Em = (p*E).sum(1)/w
            fr = np.asarray(sims[NUS[valid[0]]]['frames']); sel = fr > poke(obj, t, k)['start_frame']
            fs = np.stack([sims[NUS[i]]['force'][sel] for i in valid])            # (nv, K, 3) at E_REF
            force = np.einsum('v,v,vkj->kj', w, Em/RF.E_REF, fs)
            surf = np.einsum('v,vknj->knj', w, np.stack([sims[NUS[i]]['surf_V'][sel] for i in valid]).astype(np.float64))
            # risk: P(force RMSE > 2 N) under the posterior predictive (parameter + plug-in noise)
            flat = p.ravel(); idx = rng.choice(len(flat), N_RISK, p=flat); iv, ie = np.unravel_index(idx, p.shape)
            mu = fs[iv]*(E[ie]/RF.E_REF)[:, None, None]; y = mu+rng.normal(0, np.sqrt(s2F), mu.shape)
            risk = float((np.sqrt(((y-force)**2).sum(-1).mean(-1)) > 2.0).mean())
            meta = dict(method='ours', nu_used=[NUS[i] for i in valid], nu_weights=w.tolist(), E_mean_given_nu=Em.tolist(),
                        snapped_nu=[nu for nu in NUS if sims[nu] is not None and sims[nu]['snapped']], flag=flag,
                        degraded_frames={f'{nu:.1f}': int((sims[nu]['degraded'] > 0).sum()) for nu in NUS if sims[nu] is not None})
            save_pred('ours', obj, t, k, fr[sel], force, meta, surf_V=surf, surf_F=sims[NUS[valid[0]]]['surf_F'], risk=risk)
            # fem_lsq: point estimate; if its nu snapped on this poke, nearest non-snapped nu (recorded)
            nu_l = lsq['nu']; zl = sims[nu_l]; note = None
            if zl is None or zl['snapped']:
                alt = [nu for nu in sorted(NUS, key=lambda v: abs(v-lsq['nu'])) if sims[nu] is not None and not sims[nu]['snapped']]
                if not alt:  # every available run snapped: nearest available run
                    alt = [nu for nu in sorted(NUS, key=lambda v: abs(v-lsq['nu'])) if sims[nu] is not None]
                if alt:
                    nu_l = alt[0]; zl = sims[nu_l]; note = f"nu {lsq['nu']} failed or snapped on this poke; used nu {nu_l}"; n_fb += 1
            save_pred('fem_lsq', obj, t, k, fr[sel], zl['force'][sel]*lsq['E']/RF.E_REF,
                      dict(method='fem_lsq', nu=nu_l, E=lsq['E'], note=note), surf_V=zl['surf_V'][sel], surf_F=zl['surf_F'])
        (PRED/'ours'/obj/'missing.json').write_text(json.dumps(dict(no_prediction=missing, reason='simulation failed (timeout) at every nu')))
        print(f'{obj}: {len(heldout_pokes(obj))-len(missing)}/{len(heldout_pokes(obj))} held-out pokes predicted; fem_lsq nu fallbacks {n_fb}', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'sim': run_sims, 'status': status, 'fit': run_fit, 'predict': run_predict}[cmd]()
