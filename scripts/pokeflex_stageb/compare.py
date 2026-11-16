"""Combined PokeFlex comparison: v01 (sealed freeze) + stage B (separate, manifest-checked).  Reads only scores.json,
scores-stageB.json, fit/summary records and the catalogue stiffness; no outcomes.
Tables: (1) per-object mean force RMSE / along-approach RMSE / tool-excluded chamfer on the common poke set (pokes every
method predicted); (2) per-poke win counts vs ours; (3) stiffness ranking: Spearman vs PokeFlex stiffness_N_per_m and
number of discordant pairs, for methods that report a stiffness scalar.
usage: python -I compare.py  -> reports/pokeflex-v01/comparison.json (+ printed tables)"""
import json, itertools
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_full import verify_full
verify_full()  # freeze + code/objects/protocol hashes + pre-freeze commit; raises on any mismatch

R = Path(__file__).resolve().parents[2]
REP = R/'reports/pokeflex-v01'; PRED = R/'runs/pokeflex-v01/pred'
OBJECTS = ['FoamDice', 'FoamCylinder', 'Sponge', 'MemoryFoam', 'FoamHalfSphere', 'Pillow', '3dPrintedCylinder', '3dPrintedPyramid']
V01, SB = json.loads((REP/'scores.json').read_text()), json.loads((REP/'scores-stageB.json').read_text())
rows = V01['rows']+SB['rows']
CM = REP/'contact-metrics.json'
if CM.exists():  # post-hoc contact-region metrics (contact_metrics.py), merged per (method, poke)
    cm = {(r['method'], r['object'], r['take'], r['poke']): r for r in json.loads(CM.read_text())['rows']}
    for r in rows:
        r.update({k: v for k, v in cm.get((r['method'], r['object'], r['take'], r['poke']), {}).items()
                  if k in ('force_contact_rmse_N', 'force_contact_along_rmse_N', 'peak_force_err_N', 'peak_force_rel', 'tool_region_mm')})
METHODS = ['ours', 'fem_lsq', 'hooke', 'null', 'PhysTwin', 'PAC-NeRF']
KEYS = ['force_rmse_N', 'force_along_rmse_N', 'chamfer_excl_tool_mm', 'chamfer_mm']
KEYS += ['force_contact_rmse_N', 'force_contact_along_rmse_N', 'peak_force_err_N', 'peak_force_rel', 'tool_region_mm'] if CM.exists() else []
key = lambda r: (r['object'], r['take'], r['poke'])
by = {m: {key(r): r for r in rows if r['method'] == m} for m in METHODS}
common = set.intersection(*(set(v) for v in by.values()))


def rankdata(x):
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def spearman(a, b):
    ra, rb = rankdata(np.array(a)), rankdata(np.array(b)); return float(np.corrcoef(ra, rb)[0, 1])


table = {}
for m in METHODS:
    for o in OBJECTS:
        rr = [by[m][k] for k in common if k[0] == o]
        table.setdefault(m, {})[o] = {q: float(np.mean([r[q] for r in rr if q in r])) if any(q in r for r in rr) else float('nan')
                                      for q in KEYS} | dict(pokes=len(rr))
    allr = [by[m][k] for k in common]
    table[m]['_mean_over_objects'] = {q: float(np.nanmean([table[m][o][q] for o in OBJECTS])) for q in KEYS}
    table[m]['_pooled_pokes'] = {q: float(np.mean([r[q] for r in allr if q in r])) for q in KEYS} | dict(pokes=len(allr))

wins = {}
for m in METHODS:
    if m == 'ours':
        continue
    w = {}
    for q in KEYS:
        d = np.array([by['ours'][k][q]-by[m][k][q] for k in common if q in by['ours'][k] and q in by[m][k]])
        w[q] = dict(ours_better=int((d < 0).sum()), other_better=int((d > 0).sum()), median_diff=float(np.median(d)))
    wins[m] = w

cat = json.loads((R/'scripts/pokeflex_v01/objects.json').read_text())['objects']
truth = [cat[o]['stiffness_N_per_m'] for o in OBJECTS]
sb_sum = json.loads((R/'runs/pokeflex-v01/stageB/summary.json').read_text())
scal = {'ours': [json.loads((PRED/'ours'/o/'fit.json').read_text())['ranking_dF_dd_N_per_m'] for o in OBJECTS],
        'hooke': [json.loads((PRED/'hooke'/o/'fit.json').read_text())['k_N_per_m'] for o in OBJECTS],
        'PhysTwin': [sb_sum[o]['PhysTwin']['dFdd'] for o in OBJECTS],
        'PAC-NeRF': [sb_sum[o]['PAC-NeRF']['dFdd'] for o in OBJECTS]}
ranking = {}
for m, s in scal.items():
    disc = sum(1 for i, j in itertools.combinations(range(len(OBJECTS)), 2) if (s[i]-s[j])*(truth[i]-truth[j]) < 0)
    ranking[m] = dict(spearman=spearman(s, truth), discordant_pairs=disc, of_pairs=len(OBJECTS)*(len(OBJECTS)-1)//2,
                      scalar_N_per_m=dict(zip(OBJECTS, map(float, s))))
ranking['_catalogue_N_per_m'] = dict(zip(OBJECTS, truth))

out = dict(note='v01 methods from the sealed freeze; PhysTwin and PAC-NeRF are stage B (blind, manifest committed before '
                'the freeze) and are reported separately per the protocol amendment. Common set = pokes every method predicted.',
           common_pokes=len(common), per_object=table, ours_vs=wins, stiffness_ranking=ranking)
(REP/'comparison.json').write_text(json.dumps(out, indent=1))

print(f'common pokes: {len(common)}')
for q in KEYS:
    print(f'\n{q} (mean over pokes per object; common set)')
    print(f"{'object':18s}"+''.join(f'{m:>10s}' for m in METHODS))
    for o in OBJECTS+['_mean_over_objects']:
        print(f'{o:18s}'+''.join(f"{table[m][o][q]:10.2f}" for m in METHODS))
print('\nours vs other (per poke):')
for m, w in wins.items():
    print(f'  {m:9s}', {q: f"{v['ours_better']}-{v['other_better']} med {v['median_diff']:+.2f}" for q, v in w.items()})
print('\nstiffness ranking (Spearman, discordant/28):')
for m in scal:
    print(f"  {m:9s} {ranking[m]['spearman']:+.3f}  {ranking[m]['discordant_pairs']}")
