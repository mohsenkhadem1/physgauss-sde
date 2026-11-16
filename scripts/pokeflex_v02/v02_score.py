"""Score the PokeFlex v02 ladder on the v01 held-out pokes (post-hoc, non-blind; protocol_v02.json).

Gates (all must pass before any outcome is read):
  1. scripts/pokeflex_stageb/verify_full.py (v01 freeze, code, protocol, objects, freeze record);
  2. the stage-B manifest check (import of score_stageb), for the PhysTwin reference;
  3. reports/pokeflex-v02/MANIFEST.sha256.json must be byte-identical to its committed version (latest commit touching it)
     and to runs/pokeflex-v02/pred/MANIFEST.sha256.json; every v02 prediction file and the prediction-generating code
     (v02_model.py, v02_fit.py, protocol_v02.json) and the selection record must match it.  Changes to this scorer after
     the manifest are not refused but reported in the output (scorer_changed_since_manifest).
Metrics: pfx_score.score_poke unchanged (protocol), contact_metrics.score unchanged (contact), and the new
phase metrics below.  M0 = v01 'ours' (frozen); its protocol/contact rows are taken from the committed v01 score files
(identical code and files) after a recomputation check on a few pokes; PhysTwin likewise from the stage-B score files.
  unloading_along_rmse_N  along-approach force RMSE over scored frames strictly after the max-depth frame (PRIMARY)
  loading_along_rmse_N    scored frames up to and including the max-depth frame
  hysteresis_err_mJ       |A_pred - A_meas|, A = trapezoid sum of F_along d(depth) over consecutive scored frames
usage: python -I v02_score.py [--quick]  -> runs/pokeflex-v02/scores.json, reports/pokeflex-v02/scores.json"""
import hashlib, json, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'scripts/pokeflex_v01')); sys.path.insert(0, str(ROOT/'scripts/pokeflex_stageb'))
from verify_full import verify_full
FZ = verify_full()
import score_stageb as SB          # verifies the committed stage-B manifest at import
import contact_metrics as CM       # runs verify_full again at import; its score() is reused unchanged
import pfx_score as S
import pokeflex as P
from pfx_common import OBJECTS, approach_axis, contact_depth, heldout_pokes

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REP2 = ROOT/'reports/pokeflex-v02'; OUT2 = ROOT/'runs/pokeflex-v02'; PRED2 = OUT2/'pred'
MAN_REP, MAN_RUN = REP2/'MANIFEST.sha256.json', PRED2/'MANIFEST.sha256.json'
V01P = ROOT/'runs/pokeflex-v01/pred'
RUNGS = ['M1', 'M1b', 'M2', 'M3']
ROOTS = {'M0': V01P/'ours', 'PhysTwin': SB.SB/'PhysTwin', **{r: PRED2/r for r in RUNGS}}
FILE_METHOD = {'M0': 'ours', 'PhysTwin': 'PhysTwin', **{r: r for r in RUNGS}}


def load_v02_manifest():
    rel = MAN_REP.relative_to(ROOT).as_posix()
    commit = subprocess.run(['git', '-C', str(ROOT), 'log', '-n1', '--format=%H', '--', rel], capture_output=True, text=True, check=True).stdout.strip()
    if not commit:
        raise SystemExit('v02 manifest is not committed')
    blob = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'], capture_output=True, check=True).stdout
    if blob != MAN_REP.read_bytes() or blob != MAN_RUN.read_bytes():
        raise SystemExit('v02 manifest differs from its committed version')
    m = json.loads(blob)
    bad = [f for f, h in m['prediction_files'].items() if not (ROOT/f).exists() or sha(ROOT/f) != h]
    bad += [f for f, h in m['selection_record'].items() if sha(ROOT/f) != h]
    gen = [f for f in m['code_files'] if Path(f).name in ('v02_model.py', 'v02_fit.py', 'protocol_v02.json')]
    bad += [f for f in gen if sha(ROOT/f) != m['code_files'][f]]
    if bad:
        raise SystemExit(f'v02 files changed since manifest: {bad[:5]}')
    me = HERE.joinpath('v02_score.py').relative_to(ROOT).as_posix()
    return m, commit, sha(HERE/'v02_score.py') != m['code_files'].get(me)


MAN, MAN_COMMIT, SCORER_CHANGED = load_v02_manifest()
V02 = {str((ROOT/f).resolve()) for f in MAN['prediction_files']}
V01F = set(FZ['files'])


def sealed(f):
    f = Path(f).resolve()
    return str(f) in V02 or str(f) in SB.SEALED or f.relative_to(P.R).as_posix() in V01F


P.is_prediction_sealed = sealed
CM.sealed = sealed
CM.METHODS = {FILE_METHOD[m]: ROOTS[m] for m in ROOTS}


def phase_metrics(m, obj, take, k):
    f = ROOTS[m]/obj/f'{take}-p{k}.npz'
    if not f.exists() or not sealed(f) or not sealed(f.with_suffix('.json')):
        return None
    z = np.load(f); fr = z['frames']; a = approach_axis(obj, take)
    Fm = P.forces(obj, take, fr)[:, :3]-P.force_offset(obj, take)[:3]
    cf, dep = contact_depth(obj, take, k); dmap = dict(zip(cf.tolist(), dep.tolist()))
    d = np.array([dmap[int(x)] for x in fr]); jm = int(np.argmax(d))
    fp, fm = z['force'] @ -a, Fm @ -a; e = fp-fm
    area = lambda f: float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(d))*1e3)
    row = dict(method=m, object=obj, take=take, poke=k, loading_along_rmse_N=float(np.sqrt((e[:jm+1]**2).mean())),
               hyst_meas_mJ=area(fm), hyst_pred_mJ=area(fp))
    row['hysteresis_err_mJ'] = abs(row['hyst_pred_mJ']-row['hyst_meas_mJ'])
    if jm+1 < len(fr):
        row['unloading_along_rmse_N'] = float(np.sqrt((e[jm+1:]**2).mean())); row['unloading_frames'] = int(len(fr)-jm-1)
    if 'risk' in z.files:
        row['predicted_risk'] = float(z['risk'])
    return row


def protocol_row(args):
    m, obj, take, k = args
    S.PRED = ROOTS[m].parent
    r = S.score_poke((ROOTS[m].name, obj, take, k))
    if r:
        r['method'] = m
    return r


def contact_row(args):
    m, obj, take, k = args
    r = CM.score((FILE_METHOD[m], obj, take, k))
    if r:
        r['method'] = m
    return r


def phase_row(args):
    return phase_metrics(*args)


KEYS_P = ['force_rmse_N', 'force_along_rmse_N', 'chamfer_excl_tool_mm', 'chamfer_mm']
KEYS_C = ['tool_region_mm', 'force_contact_rmse_N', 'force_contact_along_rmse_N', 'peak_force_err_N', 'peak_force_rel']
KEYS_N = ['unloading_along_rmse_N', 'loading_along_rmse_N', 'hysteresis_err_mJ']
KEYS = KEYS_N+KEYS_P+KEYS_C


def boot_ci(diff, clusters, n=10000, seed=0):
    rng = np.random.default_rng(seed); u = sorted(set(clusters)); lab = np.array([u.index(c) for c in clusters])
    sums = np.array([diff[lab == i].sum() for i in range(len(u))]); cnt = np.array([(lab == i).sum() for i in range(len(u))])
    idx = rng.integers(0, len(u), (n, len(u)))
    b = sums[idx].sum(1)/cnt[idx].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    quick = '--quick' in sys.argv
    pokes = [(o, t, k) for o in OBJECTS for t, k in heldout_pokes(o)]
    if quick:
        pokes = pokes[::20]
    v01 = json.loads((ROOT/'reports/pokeflex-v01/scores.json').read_text())['rows']
    sb = json.loads((ROOT/'reports/pokeflex-v01/scores-stageB.json').read_text())['rows']
    cm01 = json.loads((ROOT/'reports/pokeflex-v01/contact-metrics.json').read_text())['rows']
    ref = {}
    for r in v01+sb+cm01:
        m = {'ours': 'M0', 'PhysTwin': 'PhysTwin'}.get(r['method'])
        if m:
            ref.setdefault((m, r['object'], r['take'], r['poke']), {}).update({q: r[q] for q in KEYS_P+KEYS_C if q in r})
    jobs_new = [(m, *p) for m in RUNGS for p in pokes]
    check = [('M0', *p) for p in pokes[::40]]
    with Pool(8) as pool:
        prow = pool.map(protocol_row, jobs_new+check, chunksize=1)
        crow = pool.map(contact_row, jobs_new+check, chunksize=1)
        nrow = pool.map(phase_row, [(m, *p) for m in ROOTS for p in pokes], chunksize=4)
    rows = {}
    for r in prow+crow+nrow:
        if r:
            rows.setdefault((r['method'], r['object'], r['take'], r['poke']), {}).update(r)
    recheck = []
    for m, o, t, k in check:
        new, old = rows.get((m, o, t, k), {}), ref.get((m, o, t, k), {})
        recheck.append(dict(poke=f'{t}-p{k}', maxabs=max([abs(new[q]-old[q]) for q in KEYS_P+KEYS_C if q in new and q in old] or [np.nan])))
    for key, v in ref.items():
        if key in rows:
            rows[key] = {**v, **{q: x for q, x in rows[key].items() if q not in KEYS_P+KEYS_C}}
    methods = list(ROOTS)
    pk = lambda key: key[1:]
    have = {m: {pk(k) for k, r in rows.items() if k[0] == m and 'force_rmse_N' in r and 'unloading_along_rmse_N' in r} for m in methods}
    common = sorted(set.intersection(*have.values()))
    agg = {}
    for m in methods:
        agg[m] = {}
        for o in OBJECTS:
            rr = [rows[(m, *p)] for p in common if p[0] == o]
            agg[m][o] = {q: float(np.mean([r[q] for r in rr if q in r])) for q in KEYS if any(q in r for r in rr)} | dict(pokes=len(rr))
        agg[m]['_mean_over_objects'] = {q: float(np.nanmean([agg[m][o].get(q, np.nan) for o in OBJECTS])) for q in KEYS}
        agg[m]['_pooled_pokes'] = {q: float(np.mean([rows[(m, *p)][q] for p in common if q in rows[(m, *p)]])) for q in KEYS}
    comps = {}
    prev = {'M1': 'M0', 'M2': 'M1', 'M3': 'M2', 'M1b': 'M1'}
    pairs = [(r, 'M0') for r in RUNGS]+[(r, prev[r]) for r in RUNGS if prev[r] != 'M0']+[(r, 'PhysTwin') for r in ['M0']+RUNGS]
    for a, b in pairs:
        w = {}
        for q in KEYS:
            ps_ = [p for p in common if q in rows[(a, *p)] and q in rows[(b, *p)]]
            d = np.array([rows[(a, *p)][q]-rows[(b, *p)][q] for p in ps_])
            if not len(d):
                continue
            w[q] = dict(n=len(d), a_better=int((d < 0).sum()), b_better=int((d > 0).sum()), median_diff=float(np.median(d)),
                        mean_diff=float(d.mean()), ci95_mean_diff=boot_ci(d, [(p[0], p[1]) for p in ps_]))
        comps[f'{a}_vs_{b}'] = w
    crit = {}
    for r in RUNGS:
        u = comps[f'{r}_vs_M0']['unloading_along_rmse_N']; s = comps[f'{r}_vs_M0']['chamfer_excl_tool_mm']
        dmo = agg[r]['_mean_over_objects']['chamfer_excl_tool_mm']-agg['M0']['_mean_over_objects']['chamfer_excl_tool_mm']
        crit[r] = dict(unloading_improves=u['ci95_mean_diff'][1] < 0, surface_mean_over_objects_diff_mm=dmo,
                       surface_ci_upper_mm=s['ci95_mean_diff'][1], no_surface_degradation=bool(dmo <= 0.05 and s['ci95_mean_diff'][1] <= 0.10))
    risk = {}
    for m in methods:
        rr = [rows[(m, *p)] for p in common if 'predicted_risk' in rows[(m, *p)]]
        if rr:
            pr = np.array([r['predicted_risk'] for r in rr]); obs = np.array([r['force_rmse_N'] > 2 for r in rr], float)
            risk[m] = dict(mean_pred=float(pr.mean()), observed=float(obs.mean()), brier=float(((pr-obs)**2).mean()))
    out = dict(note='v02 mechanism ladder, post-hoc and non-blind (protocol_v02.json). M0 = v01 ours (frozen); PhysTwin = stage B reference.',
               manifest_commit=MAN_COMMIT, manifest_created_utc=MAN['created_utc'], scorer_changed_since_manifest=SCORER_CHANGED,
               v01_freeze_utc=FZ['sealed_utc'], quick=quick, common_pokes=len(common), m0_recheck=recheck,
               per_object=agg, paired=comps, criteria=crit, risk=risk,
               rows=[rows[(m, *p)] for m in methods for p in common])
    name = 'scores-quick.json' if quick else 'scores.json'
    (OUT2/name).write_text(json.dumps(out, indent=1))
    if not quick:
        REP2.mkdir(exist_ok=True); (REP2/name).write_text(json.dumps(out, indent=1))
    print('common pokes', len(common), 'scorer changed:', SCORER_CHANGED, 'M0 recheck', recheck)
    for q in KEYS:
        print(f'{q:28s}'+''.join(f'{m:>10s} {agg[m]["_mean_over_objects"][q]:7.3f}' for m in methods))
    for k_, w in comps.items():
        print(k_, {q: f"{v['a_better']}-{v['b_better']} {v['mean_diff']:+.3f} [{v['ci95_mean_diff'][0]:+.3f},{v['ci95_mean_diff'][1]:+.3f}]" for q, v in w.items() if q in KEYS_N+['chamfer_excl_tool_mm', 'force_rmse_N', 'tool_region_mm']})
    print(json.dumps(crit, indent=1))


if __name__ == '__main__':
    main()
