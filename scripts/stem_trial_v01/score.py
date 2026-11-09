"""Score sealed D2 predictions against D2 truth. Verifies the freeze hashes first."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem
from ours import STATIONS

R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'; REP = R/'reports/stem-trial-v01'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
DELTA = 3e-3


def truth_stations(episode, world, tag=''):
    z = np.load(T/f'truth/{episode}-{world}{tag}.npz')
    mesh = fem.Mesh(tuple(int(c) for c in z['cells']))
    S = np.stack([mesh.region_weights(lo, hi)[0] for lo, hi in STATIONS])
    return np.einsum('sn,fnj->fsj', S, z['U']), S@z['X']


def metrics(pred, true):
    e = pred-true
    return dict(rms_mm=float(np.sqrt((e**2).sum(-1).mean())*1e3),
                final_rms_mm=float(np.sqrt((e[-1]**2).sum(-1).mean())*1e3),
                lateral_rms_mm=float(np.sqrt((e[..., 2]**2).mean())*1e3),
                vertical_rms_mm=float(np.sqrt((e[..., 1]**2).mean())*1e3),
                peak_z_075L_pred_mm=float(pred[:, 2, 2].max()*1e3), peak_z_075L_true_mm=float(true[:, 2, 2].max()*1e3),
                failure=bool(np.sqrt((e**2).sum(-1).mean()) > DELTA))


def main():
    fz = json.loads((REP/'prediction-freeze.json').read_text())
    for k, h in fz['predictions'].items():
        assert sha(T/k) == h, f'prediction changed after sealing: {k}'
    rows = []; series = {}
    gd0 = np.load(T/'gic/forecast-D0.npz')
    for w in ['W500', 'W1000', 'W2000']:
        true, ref = truth_stations('D2', w)
        series[f'truth-{w}'] = true
        o = np.load(T/f'ours/pred-{w}.npz'); oj = json.loads((T/f'ours/pred-{w}.json').read_text())
        g1 = np.load(T/f'gic/forecast-D1-{w}.npz')
        og = np.load(T/f'ours-offgrid/pred-{w}.npz'); ogj = json.loads((T/f'ours-offgrid/pred-{w}.json').read_text())
        preds = {
            ('frozen shape (zero displacement)', 'none'): np.zeros_like(true),
            ('GIC', 'D0 only'): gd0['station_centroid']-gd0['station_centroid'][0],
            ('GIC', 'after D1 poke refit'): g1['station_centroid']-g1['station_centroid'][0],
            ('GIC simulator, true E/nu/rho (oracle control)', 'D0 reconstruction'): (lambda c: c-c[0])(np.load(T/f'gic/forecast-oracle-{w}.npz')['station_centroid']),
            ('ours posterior mean', 'D0 only'): o['prediction_station_disp_D0'],
            ('ours posterior mean', 'D0 + D1 poke'): o['prediction_station_disp_D0+D1'],
            ('ours MAP', 'D0 only'): o['map_station_disp_D0'],
            ('ours MAP', 'D0 + D1 poke'): o['map_station_disp_D0+D1'],
            ('ours posterior mean, off-grid prior', 'D0 only'): og['prediction_station_disp_D0'],
            ('ours posterior mean, off-grid prior', 'D0 + D1 poke'): og['prediction_station_disp_D0+D1'],
        }
        for (m, st), p in preds.items():
            assert p.shape == true.shape, (m, st, p.shape)
            r = dict(world=w, method=m, evidence=st, **metrics(p, true))
            if m.startswith('ours posterior mean'):
                src = ogj if 'off-grid' in m else oj
                r['predicted_risk_rms_gt_3mm'] = src['D0' if st == 'D0 only' else 'D0+D1']['risk_rms_gt_3mm']
            rows.append(r); series[f'{m}|{st}|{w}'] = p
    # numerical qualification of the truth itself (available only for D1-W500 before sealing)
    qual = {}
    base, _ = truth_stations('D1', 'W500')
    for tag in ['-qual-coarse', '-qual-dt2']:
        if (T/f'truth/D1-W500{tag}.npz').exists():
            q, _ = truth_stations('D1', 'W500', tag)
            qual[tag] = float(np.sqrt(((q-base)**2).sum(-1).mean())*1e3)
    out = dict(freeze=fz['sealed_utc'], delta_mm=DELTA*1e3, rows=rows, truth_numerical_qualification_D1W500_rms_mm=qual,
               truth_sha256={w: sha(T/f'truth/D2-{w}.npz') for w in ['W500', 'W1000', 'W2000']})
    (REP/'scores.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(T/'scored-series.npz', **{k: v for k, v in series.items()})
    for r in rows:
        print(f"{r['world']:6s} {r['method']:34s} {r['evidence']:22s} rms {r['rms_mm']:7.2f}  final {r['final_rms_mm']:7.2f}  lat {r['lateral_rms_mm']:7.2f}  peak z {r['peak_z_075L_pred_mm']:6.2f}/{r['peak_z_075L_true_mm']:6.2f}" + (f"  risk {r['predicted_risk_rms_gt_3mm']:.3f}" if 'predicted_risk_rms_gt_3mm' in r else ''))
    print('truth qualification', qual)


if __name__ == '__main__':
    main()
