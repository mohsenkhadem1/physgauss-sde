"""Round-2 scoring: D2b physics (prospective), RGB-D physics on D2 (post hoc), and camera-0 rendering
for observed motion, D2 (post-hoc metric on sealed predictions) and D2b (prospective).
Verifies prediction-freeze-v2 hashes first. Run with pacnerf-env python."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import truth_stations, metrics

R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'; REP = R/'reports/stem-trial-v01'
W = ['W500', 'W1000', 'W2000']
KEEP = [12, 24, 36]  # frames kept for figures; metrics use all frames
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
disp = lambda c: c-c[0]


def physics():
    rows = []; series = {}
    for w in W:
        for ep in ('D2b', 'D2'):
            true, _ = truth_stations(ep, w); series[f'truth|{ep}|{w}'] = true
            r2 = lambda d, s: np.load(T/f'{d}/round2-{s}-{w}.npz')
            preds = {('ours RGB-D', 'D0 only'): r2('ours', 'rgbd')[f'pred_{ep}_D0'],
                     ('ours RGB-D', 'D0 + D1 poke'): r2('ours', 'rgbd')[f'pred_{ep}_D0+D1']}
            if ep == 'D2b':
                preds.update({
                    ('frozen shape (zero displacement)', 'none'): np.zeros_like(true),
                    ('GIC', 'D0 only'): disp(np.load(T/'gic/forecast-D2b-D0.npz')['station_centroid']),
                    ('GIC', 'after D1 poke refit'): disp(np.load(T/f'gic/forecast-D2b-D1-{w}.npz')['station_centroid']),
                    ('GIC simulator, true E/nu/rho (oracle control)', 'D0 reconstruction'): disp(np.load(T/f'gic/forecast-D2b-oracle-{w}.npz')['station_centroid']),
                    ('ours tracks', 'D0 only'): r2('ours', 'tracks')['pred_D2b_D0'],
                    ('ours tracks', 'D0 + D1 poke'): r2('ours', 'tracks')['pred_D2b_D0+D1'],
                    ('ours tracks, off-grid prior', 'D0 only'): r2('ours-offgrid', 'tracks')['pred_D2b_D0'],
                    ('ours tracks, off-grid prior', 'D0 + D1 poke'): r2('ours-offgrid', 'tracks')['pred_D2b_D0+D1']})
            for (m, st), p in preds.items():
                r = dict(episode=ep, status='prospective' if ep == 'D2b' else 'post hoc (D2 opened in round 1)',
                         world=w, method=m, evidence=st, **metrics(p, true))
                if m.startswith('ours'):
                    d = 'ours-offgrid' if 'off-grid' in m else 'ours'; s = 'rgbd' if 'RGB-D' in m else 'tracks'
                    r['predicted_risk_rms_gt_3mm'] = json.loads((T/f'{d}/round2-{s}-{w}.json').read_text())['D0' if st == 'D0 only' else 'D0+D1'][f'risk_{ep}']
                rows.append(r); series[f'{m}|{st}|{ep}|{w}'] = p
    return rows, series


def ref_images(scene):
    ims = []
    for f in range(len(list((T/f'scenes/{scene}/data').glob('m_0_*.png')))-1):
        im = np.asarray(Image.open(T/f'scenes/{scene}/data/m_0_{f}.png')).astype(float)/255
        ims.append(im)
    return np.stack(ims)


def to_black(ims_u8):
    """GIC reader rule: pixels that are not pure white are foreground; background composited to black."""
    m = ims_u8.astype(int).sum(-1, keepdims=True) != 765
    return ims_u8/255*m


def img_metrics(pred, ref):
    """PSNR from MSE pooled over frames 1..end (frame 0 is the shared, given rest state) on the
    reference foreground mask, and full-frame; both after the GIC reader's black compositing."""
    ref_u8 = np.round(ref*255).astype(np.uint8); ref_b = to_black(ref_u8); mask = ref_u8.astype(int).sum(-1) != 765
    e = (pred-ref_b)**2
    fg = float(-10*np.log10(np.mean([e[f][mask[f]].mean() for f in range(1, len(e))])))
    full = float(-10*np.log10(e[1:].mean()))
    per = [float(-10*np.log10(e[f][mask[f]].mean())) for f in range(1, len(e))]
    return dict(fg_psnr_db=fg, full_psnr_db=full, fg_psnr_per_frame=per)


def rendering():
    rows = []; panels = {}
    load = lambda p: np.load(p)['images'].astype(float)/255
    refs = {s: ref_images(s) for s in ['D0-W1000']+[f'{e}-{w}' for e in ('D1', 'D2', 'D2b') for w in W] if (T/f'scenes/{s}').exists()}
    def add(label, appearance, world, episode, status, pred, ref_key):
        r = dict(method=label, appearance=appearance, world=world, episode=episode, status=status, **img_metrics(pred, refs[ref_key]))
        rows.append(r); panels[f'{appearance}|{label}|{episode}|{world}'] = (pred[[f for f in KEEP if f < len(pred)]]*255).round().astype(np.uint8)
    add('GIC deformation network', 'GIC D0', 'all', 'D0 observed', 'fit', load(T/'renders/gic-D0/deformnet-fit.npz'), 'D0-W1000')
    add('GIC physics', 'GIC D0', 'all', 'D0 observed', 'fit', load(T/'renders/gic-D0/physics-fit.npz'), 'D0-W1000')
    for w in W:
        add('GIC deformation network', f'GIC D1-{w}', w, 'D1 observed', 'fit', load(T/f'renders/gic-D1-{w}/deformnet-fit.npz'), f'D1-{w}')
        add('GIC physics', f'GIC D1-{w}', w, 'D1 observed', 'fit', load(T/f'renders/gic-D1-{w}/physics-fit.npz'), f'D1-{w}')
    for f in sorted((T/'renders/shared').glob('*.npz')):
        model, ep, w, label = f.stem.split('|')
        st = 'prospective' if ep == 'D2b' else 'post-hoc metric on sealed predictions'
        add(label, f'GIC {model}', w, ep, st, load(f), f'{ep}-{w}')
    return rows, panels, refs


def main():
    fz = json.loads((REP/'prediction-freeze-v2.json').read_text())
    for k, h in fz['files'].items():
        assert sha(T/k) == h, f'changed after sealing: {k}'
    prow, series = physics(); rrow, panels, refs = rendering()
    (REP/'scores-round2.json').write_text(json.dumps(dict(freeze=fz['sealed_utc'], physics=prow, rendering=[{k: v for k, v in r.items() if k != 'fg_psnr_per_frame'} for r in rrow],
                                                          rendering_per_frame={f"{r['method']}|{r['episode']}|{r['world']}": r['fg_psnr_per_frame'] for r in rrow}), indent=1))
    np.savez_compressed(T/'scored-series-round2.npz', **series)
    np.savez_compressed(T/'render-panels.npz', **panels,
                        **{f'ref|{k}': (to_black(np.round(v[[f for f in KEEP if f < len(v)]]*255).astype(np.uint8))*255).round().astype(np.uint8) for k, v in refs.items()})
    for r in prow:
        print(f"{r['episode']:4s} {r['world']:6s} {r['method'][:40]:40s} {r['evidence']:20s} rms {r['rms_mm']:6.2f} lat {r['lateral_rms_mm']:6.2f} vert {r['vertical_rms_mm']:6.2f}" + (f" risk {r['predicted_risk_rms_gt_3mm']:.3f}" if 'predicted_risk_rms_gt_3mm' in r else ''))
    for r in rrow:
        print(f"RENDER {r['appearance']:13s} {r['episode']:12s} {r['world']:6s} {r['method'][:28]:28s} fg {r['fg_psnr_db']:6.2f} full {r['full_psnr_db']:6.2f}")


if __name__ == '__main__':
    main()
