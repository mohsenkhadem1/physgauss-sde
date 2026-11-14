"""Figure-1 freeze and scoring.
freeze SCEN   -> reports/fig1-v01/prediction-freeze-SCEN.json (refuses if held-out truth exists)
score         -> reports/fig1-v01/fig1-scores.json (verifies every freeze first; scores what is sealed)
SCEN: S1 (stem push D2c), S2 (block press D2), S3 (stem striker D3)."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
ST = R/'runs/stem-trial-v01'; BK = R/'runs/fig1-v01/block'; REP = R/'reports/fig1-v01'
SW = ['W500', 'W1000', 'W2000']; BW = ['Bleft', 'Bcentre', 'Bright']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
STR_M, STR_V0 = 0.3, -1.0
EXT = R/'runs/fig1-v01'; EXTERNAL = ('mphygs', 'phystwin')  # adapters of M-PhyGs and PhysTwin (forecast-{EP}-{D0|D1}-{world})
EXT_NAME = {'mphygs': 'M-PhyGs', 'phystwin': 'PhysTwin'}


def files(scen):
    f = {}
    if scen == 'S1':
        for w in SW:
            for s in ('rgb', 'tracks', 'rgbd'):
                for ext in ('npz', 'json'):
                    f[f'ours/round3-{s}-{w}.{ext}'] = ST/f'ours/round3-{s}-{w}.{ext}'
            f[f'gic/forecast-D2c-D1-{w}.npz'] = ST/f'gic/forecast-D2c-D1-{w}.npz'
            f[f'gic/forecast-D2c-oracle-{w}.npz'] = ST/f'gic/forecast-D2c-oracle-{w}.npz'
            f[f'pac/forecast-D2c-D1-{w}.npz'] = ST/f'pac/forecast-D2c-D1-{w}.npz'
        f['gic/forecast-D2c-D0.npz'] = ST/'gic/forecast-D2c-D0.npz'; f['pac/forecast-D2c-D0.npz'] = ST/'pac/forecast-D2c-D0.npz'
        root = ST; truth = [ST/f'truth/D2c-{w}.npz' for w in SW]
    elif scen == 'S2':
        for w in BW:
            for ext in ('npz', 'json'):
                f[f'ours/pred-rgb-{w}.{ext}'] = BK/f'ours/pred-rgb-{w}.{ext}'
                f[f'control/pred-fem-{w}.{ext}'] = BK/f'control/pred-fem-{w}.{ext}'
            for m in ('D0', 'D1'):
                f[f'gic/forecast-D2-{m}-{w}.npz'] = BK/f'gic/forecast-D2-{m}-{w}.npz'
                f[f'pac/forecast-D2-{m}-{w}.npz'] = BK/f'pac/forecast-D2-{m}-{w}.npz'
        root = BK; truth = [BK/f'truth/D2-{w}.npz' for w in BW]
    else:
        for w in SW:
            for st in ('D0', 'D0pD1'):
                for ext in ('npz', 'json'):
                    f[f'ours/striker/rgb-{w}-{st}.{ext}'] = ST/f'ours/striker/rgb-{w}-{st}.{ext}'
            f[f'gic/forecast-D3-D1-{w}.npz'] = ST/f'gic/forecast-D3-D1-{w}.npz'
            f[f'gic/forecast-D3-oracle-{w}.npz'] = ST/f'gic/forecast-D3-oracle-{w}.npz'
            f[f'pac/forecast-D3-D1-{w}.npz'] = ST/f'pac/forecast-D3-D1-{w}.npz'
        f['gic/forecast-D3-D0.npz'] = ST/'gic/forecast-D3-D0.npz'; f['pac/forecast-D3-D0.npz'] = ST/'pac/forecast-D3-D0.npz'
        root = ST; truth = [ST/f'truth/D3-{w}.npz' for w in SW]
    ep = {'S1': 'D2c', 'S2': 'D2', 'S3': 'D3'}[scen]
    for w in (BW if scen == 'S2' else SW):
        for meth in EXTERNAL:
            for st in ('D0', 'D1'):
                for ext in ('npz', 'json'):
                    p = EXT/meth/f'forecast-{ep}-{st}-{w}.{ext}'
                    f[str(p.relative_to(R))] = p
    for w in (BW if scen == 'S2' else SW):  # ours full-shape predictions (figure rows), sealed with the scores
        if scen == 'S1':
            f[f'shape-{w}'] = ST/f'ours-fields/round3-rgb-{w}-D2c.npz'
        elif scen == 'S2':
            f[f'shape-{w}'] = BK/f'ours/field-rgb-{w}.npz'
        else:
            for st in ('D0', 'D0pD1'):
                f[f'shape-{w}-{st}'] = ST/f'ours/striker/field-rgb-{w}-{st}.npz'
    f = {str(p.relative_to(R)): p for p in f.values()}  # every key relative to the repository root
    return f, R, truth


def freeze(scen):
    f, root, truth = files(scen)
    assert not any(t.exists() for t in truth), 'held-out truth already exists'
    missing = [k for k, p in f.items() if not p.exists()]
    code = {p.name: sha(p) for p in sorted(list(HERE.glob('*.py'))+list((R/'scripts/stem_trial_v01').glob('*.py')))}
    rec = dict(scenario=scen, sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files={k: sha(p) for k, p in f.items() if p.exists()},
               missing=missing, protocol_sha256=sha(HERE/'protocol_fig1.json'), code_sha256=code)
    out = REP/f'prediction-freeze-{scen}.json'; assert not out.exists(); REP.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1)); print(scen, 'sealed', len(rec['files']), 'missing', missing)


def verify(scen):
    """Check every sealed hash and return a lookup that yields a path ONLY for sealed files:
    anything recorded as missing at freeze time (or never listed) is never scored, even if it exists now."""
    fz = json.loads((REP/f'prediction-freeze-{scen}.json').read_text()); _, root, _ = files(scen)
    for k, h in fz['files'].items():
        assert sha(root/k) == h, f'{scen}: changed after sealing: {k}'
    def sealed(path):
        k = Path(path).relative_to(root).as_posix()
        return Path(path) if k in fz['files'] else None
    return sealed


def rms_mm(p, t):
    return float(np.sqrt(((p-t)**2).sum(-1).mean())*1e3)


def stem_truth_stations(ep, w):
    from score import truth_stations
    return truth_stations(ep, w)[0]


def particle_station_disp(npz):
    c = np.load(npz)['station_centroid']; return c-c[0]


def block_probe_disp_particles(npz, probes, radius=0.01):
    xyz = np.load(npz)['xyz']; ref = xyz[0]
    out = []
    for p in probes:
        sel = np.linalg.norm(ref-p, axis=1) < radius
        if not sel.any():
            sel = np.argsort(np.linalg.norm(ref-p, axis=1))[:8]
        out.append((xyz[:, sel]-ref[sel]).mean(1))
    return np.stack(out, 1)  # (F,P,3)


def score():
    import block as B, fem
    rows = []
    # S1
    if (REP/'prediction-freeze-S1.json').exists() and all((ST/f'truth/D2c-{w}.npz').exists() for w in SW):
        sealed = verify('S1')
        for w in SW:
            t = stem_truth_stations('D2c', w)
            preds = {('GIC', 'passive only'): ST/'gic/forecast-D2c-D0.npz', ('GIC', 'after poke'): ST/f'gic/forecast-D2c-D1-{w}.npz',
                     ('PAC-NeRF', 'passive only'): ST/'pac/forecast-D2c-D0.npz', ('PAC-NeRF', 'after poke'): ST/f'pac/forecast-D2c-D1-{w}.npz',
                     ('GIC simulator, true parameters', 'oracle'): ST/f'gic/forecast-D2c-oracle-{w}.npz'}
            for meth in EXTERNAL:
                preds[(EXT_NAME[meth], 'passive only')] = EXT/meth/f'forecast-D2c-D0-{w}.npz'; preds[(EXT_NAME[meth], 'after poke')] = EXT/meth/f'forecast-D2c-D1-{w}.npz'
            for (m, st), p in preds.items():
                if sealed(p):
                    rows.append(dict(scenario='S1', world=w, method=m, evidence=st, rms_mm=rms_mm(particle_station_disp(p), t)))
            for s, lab in (('rgb', 'ours (RGB)'), ('rgbd', 'ours (RGB-D)'), ('tracks', 'ours (supplied tracks)')):
                zp, jp = sealed(ST/f'ours/round3-{s}-{w}.npz'), sealed(ST/f'ours/round3-{s}-{w}.json')
                if not (zp and jp):
                    continue
                z = np.load(zp); j = json.loads(jp.read_text())
                for stage, st in (('D0', 'passive only'), ('D0+D1', 'after poke')):
                    rows.append(dict(scenario='S1', world=w, method=lab, evidence=st, rms_mm=rms_mm(z[f'pred_D2c_{stage}'], t),
                                     predicted_risk=j[stage]['risk_D2c']))
    # S2
    if (REP/'prediction-freeze-S2.json').exists() and all((BK/f'truth/D2-{w}.npz').exists() for w in BW):
        sealed = verify('S2')
        for w in BW:
            z = np.load(BK/f'truth/D2-{w}.npz'); m = B.mesh(tuple(z['cells']))
            t = np.einsum('pn,fnj->fpj', m.interpolation(B.PROBES).toarray(), z['U'])
            for meth, root, pat in (('GIC', BK/'gic', 'forecast-D2-{mm}-{w}.npz'), ('PAC-NeRF', BK/'pac', 'forecast-D2-{mm}-{w}.npz'),
                                    *((EXT_NAME[e], EXT/e, 'forecast-D2-{mm}-{w}.npz') for e in EXTERNAL)):
                for mm, st in (('D0', 'passive only'), ('D1', 'after poke')):
                    p = root/pat.format(mm=mm, w=w)
                    if sealed(p):
                        rows.append(dict(scenario='S2', world=w, method=meth, evidence=st, rms_mm=rms_mm(block_probe_disp_particles(p, B.PROBES), t)))
            cp = sealed(BK/f'control/pred-fem-{w}.npz')
            if cp and sealed(BK/f'control/pred-fem-{w}.json'):
                c = np.load(cp)
                for stage, st in (('D0', 'passive only'), ('D0+D1', 'after poke')):
                    rows.append(dict(scenario='S2', world=w, method='fitted spatially varying FEM (control)', evidence=st, rms_mm=rms_mm(c[f'pred_D2_{stage}'], t)))
            op, jp = sealed(BK/f'ours/pred-rgb-{w}.npz'), sealed(BK/f'ours/pred-rgb-{w}.json')
            if not (op and jp):
                continue
            o = np.load(op); j = json.loads(jp.read_text())
            for stage, st in (('D0', 'passive only'), ('D0+D1', 'after poke')):
                rows.append(dict(scenario='S2', world=w, method='ours (RGB)', evidence=st, rms_mm=rms_mm(o[f'pred_D2_{stage}'], t),
                                 predicted_risk=j[stage]['risk_rms_gt_3mm']))
    # S3
    if (REP/'prediction-freeze-S3.json').exists() and all((ST/f'truth/D3-{w}.npz').exists() for w in SW):
        sealed = verify('S3')
        from ours import station_matrix
        for w in SW:
            z = np.load(ST/f'truth/D3-{w}.npz'); m = fem.Mesh(tuple(z['cells'])); t = np.einsum('sn,fnj->fsj', station_matrix(m), z['U'])
            imp_t = STR_M*(z['sphere_v'][-1][2]-STR_V0)
            preds = {('GIC', 'passive only'): ST/'gic/forecast-D3-D0.npz', ('GIC', 'after poke'): ST/f'gic/forecast-D3-D1-{w}.npz',
                     ('PAC-NeRF', 'passive only'): ST/'pac/forecast-D3-D0.npz', ('PAC-NeRF', 'after poke'): ST/f'pac/forecast-D3-D1-{w}.npz',
                     ('GIC simulator, true parameters', 'oracle'): ST/f'gic/forecast-D3-oracle-{w}.npz'}
            for meth in EXTERNAL:
                preds[(EXT_NAME[meth], 'passive only')] = EXT/meth/f'forecast-D3-D0-{w}.npz'; preds[(EXT_NAME[meth], 'after poke')] = EXT/meth/f'forecast-D3-D1-{w}.npz'
            for (mth, st), p in preds.items():
                if sealed(p):
                    d = np.load(p); h = d['striker']; imp = STR_M*(h[-1][6]-STR_V0)
                    rows.append(dict(scenario='S3', world=w, method=mth, evidence=st, rms_mm=rms_mm(particle_station_disp(p), t),
                                     impulse_err_Ns=float(abs(imp-imp_t)), impulse_pred_Ns=float(imp), impulse_true_Ns=float(imp_t)))
            for stage, st in (('D0', 'passive only'), ('D0pD1', 'after poke')):
                op, jp = sealed(ST/f'ours/striker/rgb-{w}-{stage}.npz'), sealed(ST/f'ours/striker/rgb-{w}-{stage}.json')
                if not (op and jp):
                    continue
                o = np.load(op); j = json.loads(jp.read_text())
                rows.append(dict(scenario='S3', world=w, method='ours (RGB)', evidence=st, rms_mm=rms_mm(o['station_disp'], t),
                                 impulse_err_Ns=float(abs(j['impulse_mean_Ns']-imp_t)), impulse_pred_Ns=j['impulse_mean_Ns'], impulse_true_Ns=float(imp_t),
                                 predicted_risk=j['risk_station_rms_gt_3mm']))
    (REP/'fig1-scores.json').write_text(json.dumps(dict(rows=rows), indent=1))
    for r in rows:
        print(r['scenario'], f"{r['world']:8s} {r['method']:32s} {r['evidence']:13s} rms {r['rms_mm']:6.2f}" + (f"  impulse err {r['impulse_err_Ns']:.3f} ({r['impulse_pred_Ns']:.3f}/{r['impulse_true_Ns']:.3f})" if 'impulse_err_Ns' in r else ''))


if __name__ == '__main__':
    freeze(sys.argv[2]) if sys.argv[1] == 'freeze' else score()
