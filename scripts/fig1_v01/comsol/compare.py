"""Compare COMSOL re-solves of the Figure-1 reference scenarios with the in-house FEM truth.
truth  -> runs/fig1-v01/comsol/truth-scored.npz  (scored quantities exactly as fig1_score.py computes them)
compare-> prints/writes runs/fig1-v01/comsol/comparison.json from runs/fig1-v01/comsol/<case>.txt
COMSOL output format (written by the Java runners): lines 'key i j ... : values' -- see parse()."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[2]
sys.path.insert(0, str(R/'scripts/stem_trial_v01')); sys.path.insert(0, str(R/'scripts/fig1_v01'))
import fem
from ours import station_matrix
OUT = R/'runs/fig1-v01/comsol'; ST = R/'runs/stem-trial-v01/truth'; BK = R/'runs/fig1-v01/block/truth'
SW = ['W500', 'W1000', 'W2000']; BW = ['Bleft', 'Bcentre', 'Bright']


def truth():
    import block as B
    d = {}
    for w in SW:
        for ep in ('D2c', 'D3'):
            z = np.load(ST/f'{ep}-{w}.npz'); m = fem.Mesh(tuple(int(c) for c in z['cells']))
            d[f'{ep}-{w}'] = np.einsum('sn,fnj->fsj', station_matrix(m), z['U'])
            if ep == 'D3':
                d[f'{ep}-{w}-impulse'] = np.array(0.3*(z['sphere_v'][-1][2]+1.0))
                d[f'{ep}-{w}-sphere_c'] = z['sphere_c']; d[f'{ep}-{w}-sphere_v'] = z['sphere_v']
    for w in BW:
        z = np.load(BK/f'D2-{w}.npz'); m = B.mesh(tuple(z['cells']))
        d[f'D2-{w}'] = np.einsum('pn,fnj->fpj', m.interpolation(B.PROBES).toarray(), z['U'])
    np.savez(OUT/'truth-scored.npz', **d)
    for k, v in d.items():
        print(k, v.shape, f'max|.| {np.abs(v).max()*1e3:.3f} mm' if v.ndim == 3 else v)


def load(case):
    """Java writes one line per frame: 't q1x q1y q1z q2x ...' (+ sphere cz czt for D3)."""
    a = np.loadtxt(OUT/f'{case}.txt', ndmin=2)
    return a


def rms_mm(p, t):
    return float(np.sqrt(((p-t)**2).sum(-1).mean())*1e3)


def compare(cases):
    T = np.load(OUT/'truth-scored.npz'); res = {}
    for case in cases:
        key = case.split('_')[0]  # e.g. D2c-W1000_h1 -> D2c-W1000
        a = load(case); t = T[key]; nq = t.shape[1]
        q = a[:, 1:1+3*nq].reshape(len(a), nq, 3)
        assert len(a) == len(t), (case, len(a), len(t))
        r = dict(rms_diff_mm=rms_mm(q, t), max_diff_mm=float(np.linalg.norm(q-t, axis=-1).max()*1e3),
                 truth_rms_mm=float(np.sqrt((t**2).sum(-1).mean())*1e3), truth_max_mm=float(np.linalg.norm(t, axis=-1).max()*1e3))
        if key.startswith('D3'):
            czt = a[-1, 1+3*nq+5]; imp = 0.3*(czt+1.0)
            r.update(impulse_comsol_Ns=float(imp), impulse_truth_Ns=float(T[key+'-impulse']), impulse_diff_Ns=float(imp-T[key+'-impulse']),
                     sphere_cz_end_diff_mm=float((a[-1, 1+3*nq+2]-T[key+'-sphere_c'][-1][2])*1e3),
                     sphere_v_end_diff=(a[-1, 1+3*nq+3:1+3*nq+6]-T[key+'-sphere_v'][-1]).round(5).tolist())
        res[case] = r
        print(case, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
    return res


def pair(c1, c2):
    a, b = load(c1), load(c2); n = 3*(27 if c1.startswith('D2-') else 4)  # scored points only (not the striker columns)
    qa, qb = a[:, 1:1+n].reshape(len(a), -1, 3), b[:, 1:1+n].reshape(len(b), -1, 3)
    return rms_mm(qa, qb)


if __name__ == '__main__':
    if sys.argv[1] == 'truth':
        truth()
    elif sys.argv[1] == 'pair':
        print(sys.argv[2], sys.argv[3], 'rms diff mm', pair(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == 'compare':
        res = compare(sys.argv[2:]); p = OUT/'comparison.json'
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(res); p.write_text(json.dumps(old, indent=1))


def table():
    """Markdown table of every COMSOL case vs truth, with the scored method-error scale."""
    c = json.loads((OUT/'comparison.json').read_text())
    sc = json.loads((R/'reports/fig1-v01/fig1-scores.json').read_text())['rows']
    ep2s = {'D2c': 'S1', 'D2': 'S2', 'D3': 'S3'}
    print('| case | tets | order | rtol | solve s | RMS diff mm | max diff mm | truth RMS mm | ours mm | best other mm | impulse diff N s | ours / best other imp. err |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, r in c.items():
        if 'smoke' in k:
            continue
        key = k.split('_')[0]; ep, w = key.split('-'); s = ep2s[ep]
        meta = json.loads((OUT/f'{k}.json').read_text())
        rows = [x for x in sc if x['scenario'] == s and x['world'] == w and x['evidence'] in ('after poke', 'oracle')]
        ours = [x for x in rows if x['method'].startswith('ours')]; oth = [x for x in rows if not x['method'].startswith('ours') and 'control' not in x['method']]
        bo = min(oth, key=lambda x: x['rms_mm'])
        imp = f"{r['impulse_diff_Ns']:+.4f}" if 'impulse_diff_Ns' in r else ''
        impm = f"{ours[0]['impulse_err_Ns']:.3f} / {min(x['impulse_err_Ns'] for x in oth):.3f}" if 'impulse_diff_Ns' in r else ''
        print(f"| {k} | {meta['tets']} | {meta['order']} | {meta['rtol']:g} | {meta['solve_s']:.0f} | {r['rms_diff_mm']:.3f} | {r['max_diff_mm']:.3f} | {r['truth_rms_mm']:.1f} | "
              f"{ours[0]['rms_mm']:.2f} | {bo['rms_mm']:.2f} ({bo['method']}) | {imp} | {impm} |")


if __name__ == '__main__' and sys.argv[1] == 'table':
    table()


def rescore(best):
    """Re-score every sealed 'after poke'/oracle prediction against the COMSOL solution (best = {truth key: comsol case}),
    reading the predictions exactly as fig1_score.score() does. Returns rows with rms vs truth and vs COMSOL."""
    import fig1_score as S, block as B
    T = np.load(OUT/'truth-scored.npz'); rows = []
    def add(scen, w, meth, st, p, key, imp=None):
        a = load(best[key]); nq = T[key].shape[1]; q = a[:, 1:1+3*nq].reshape(len(a), nq, 3)
        r = dict(scenario=scen, world=w, method=meth, evidence=st, rms_truth=rms_mm(p, T[key]), rms_comsol=rms_mm(p, q))
        if imp is not None:
            ic = 0.3*(a[-1, 1+3*nq+5]+1.0)
            r.update(imp_err_truth=abs(imp-float(T[key+'-impulse'])), imp_err_comsol=abs(imp-ic))
        rows.append(r)
    for w in SW:
        key = f'D2c-{w}'
        if key in best:
            for meth, f in (('GIC', S.ST/f'gic/forecast-D2c-D1-{w}.npz'), ('PAC-NeRF', S.ST/f'pac/forecast-D2c-D1-{w}.npz'),
                            ('GIC oracle', S.ST/f'gic/forecast-D2c-oracle-{w}.npz'),
                            ('M-PhyGs', S.EXT/f'mphygs/forecast-D2c-D1-{w}.npz'), ('PhysTwin', S.EXT/f'phystwin/forecast-D2c-D1-{w}.npz')):
                if f.exists():
                    add('S1', w, meth, 'after poke', S.particle_station_disp(f), key)
            z = np.load(S.ST/f'ours/round3-rgb-{w}.npz'); add('S1', w, 'ours (RGB)', 'after poke', z['pred_D2c_D0+D1'], key)
        key = f'D3-{w}'
        if key in best:
            for meth, f in (('GIC', S.ST/f'gic/forecast-D3-D1-{w}.npz'), ('PAC-NeRF', S.ST/f'pac/forecast-D3-D1-{w}.npz'),
                            ('GIC oracle', S.ST/f'gic/forecast-D3-oracle-{w}.npz'),
                            ('M-PhyGs', S.EXT/f'mphygs/forecast-D3-D1-{w}.npz'), ('PhysTwin', S.EXT/f'phystwin/forecast-D3-D1-{w}.npz')):
                if f.exists():
                    d = np.load(f); add('S3', w, meth, 'after poke', S.particle_station_disp(f), key, imp=0.3*(d['striker'][-1][6]+1.0))
            o = np.load(S.ST/f'ours/striker/rgb-{w}-D0pD1.npz'); j = json.loads((S.ST/f'ours/striker/rgb-{w}-D0pD1.json').read_text())
            add('S3', w, 'ours (RGB)', 'after poke', o['station_disp'], key, imp=j['impulse_mean_Ns'])
    for w in BW:
        key = f'D2-{w}'
        if key in best:
            for meth, f in (('GIC', S.BK/f'gic/forecast-D2-D1-{w}.npz'), ('PAC-NeRF', S.BK/f'pac/forecast-D2-D1-{w}.npz'),
                            ('M-PhyGs', S.EXT/f'mphygs/forecast-D2-D1-{w}.npz'), ('PhysTwin', S.EXT/f'phystwin/forecast-D2-D1-{w}.npz')):
                if f.exists():
                    add('S2', w, meth, 'after poke', S.block_probe_disp_particles(f, B.PROBES), key)
            o = np.load(S.BK/f'ours/pred-rgb-{w}.npz'); add('S2', w, 'ours (RGB)', 'after poke', o['pred_D2_D0+D1'], key)
    for r in rows:
        print(r['scenario'], r['world'].ljust(8), r['method'].ljust(12), f"vs truth {r['rms_truth']:6.2f}  vs COMSOL {r['rms_comsol']:6.2f}" +
              (f"   imp err truth {r['imp_err_truth']:.4f} COMSOL {r['imp_err_comsol']:.4f}" if 'imp_err_truth' in r else ''))
    (OUT/'rescore.json').write_text(json.dumps(dict(best=best, rows=rows), indent=1))
    return rows


if __name__ == '__main__' and sys.argv[1] == 'rescore':
    rescore(dict(a.split('=') for a in sys.argv[2:]))
