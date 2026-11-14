"""Figure-1 compute cost per method and stage, from each method's own completion records (not the polling log, which
also holds runs killed by the GPU faults). Writes reports/fig1-v01/compute-cost.json and prints a table.
Sources:
  GIC        fit/refit: wall time from the run directory (cfg_args written at start -> 0-pred.json at end); forecast:
             runs/fig1-v01/timings.jsonl (gic_stem_predict.py). Cross-checked: block refits agree with the log to <1 min.
  PAC-NeRF   fit/refit: fit.json 'seconds'; forecast: timings.jsonl (pacnerf_stem.py forecast).
  M-PhyGs    fit: sum of params.json 'seconds' over the curriculum stages that produced the sealed checkpoint; forecast json.
  PhysTwin   fit: forecast json fit.wall_seconds (CMA + Adam); forecast json 'seconds'.
  FEM ctrl   pred-fem json 'seconds' (both fits; CPU, 13 processes).
  ours       one-off simulation bank (bank*.json 'seconds', CPU), RGB marker tracking and posterior inference re-timed on
             2026-11-14 (outputs to scratch; inference reproduces every sealed prediction bit-for-bit), striker stage json.
Conditions: RTX 2080 Ti; GPU jobs mostly shared a card with one other job, and two cards showed hardware thermal
slowdown before the cooling fix (2026-11-10/11), so GPU wall times are upper bounds. CPU jobs: 36-core host."""
import glob, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fig1_score as FS
ST, BK, EXT = FS.ST, FS.BK, FS.EXT
OURS_TIMING = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # re-timing json from the scratch run
LOG = [json.loads(l) for l in open(R/'runs/fig1-v01/timings.jsonl')]


def log_minutes(pattern):
    v = [r['wall_s'] for r in LOG if re.search(pattern, r['cmd'])]
    return float(np.median(v))/60 if v else None


def gic_dir_minutes(d):
    d = Path(d); return ((d/'0-pred.json').stat().st_mtime-(d/'cfg_args').stat().st_mtime)/60


def mphygs_fit_minutes(scene, ep, world, stages):
    tot = 0.0
    for n in stages:
        p = glob.glob(str(EXT/f'mphygs/{scene}/{ep}-{world}/run/*/alternating_photo_opt_{n}frames_bs{n}_grid75_uniform_gravity/tracking_00_mat/chunk_00000/params.json'))
        assert len(p) == 1, (scene, ep, world, n, p); tot += json.load(open(p[0]))['seconds']
    return tot/60


def stats(v):
    v = [x for x in v if x is not None]; return dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)), n=len(v)) if v else None


rows = {}
# ---- S2 block (fits per world)
B = {}
B['GIC'] = dict(fit_initial=stats([gic_dir_minutes(BK/f'gic/D0-{w}') for w in FS.BW]), refit_calibration=stats([gic_dir_minutes(BK/f'gic/D1-{w}') for w in FS.BW]),
                forecast=stats([log_minutes(rf'gic_stem_predict.py \S*block/gic/D[01]-{w} \S*forecast')for w in FS.BW]))
B['PAC-NeRF'] = dict(fit_initial=stats([json.load(open(BK/f'pac/D0-{w}/fit.json'))['seconds']/60 for w in FS.BW]),
                     refit_calibration=stats([json.load(open(BK/f'pac/D1-{w}/fit.json'))['seconds']/60 for w in FS.BW]),
                     forecast=stats([log_minutes(rf'pacnerf_stem.py forecast \S*block/pac/D1-{w}') for w in FS.BW]))
B['M-PhyGs'] = dict(fit_initial=stats([mphygs_fit_minutes('block', 'D0', w, (12, 24)) for w in FS.BW]),
                    refit_calibration=stats([mphygs_fit_minutes('block', 'D1', w, (12, 24)) for w in FS.BW]),
                    forecast=stats([json.load(open(EXT/f'mphygs/forecast-D2-D1-{w}.json'))['seconds']/60 for w in FS.BW]))
B['PhysTwin'] = dict(fit_initial=stats([json.load(open(EXT/f'phystwin/forecast-D2-D0-{w}.json'))['fit']['wall_seconds']/60 for w in FS.BW]),
                     refit_calibration=stats([json.load(open(EXT/f'phystwin/forecast-D2-D1-{w}.json'))['fit']['wall_seconds']/60 for w in FS.BW]),
                     forecast=stats([json.load(open(EXT/f'phystwin/forecast-D2-D1-{w}.json'))['seconds']/60 for w in FS.BW]))
B['FEM control'] = dict(fit_initial_plus_refit=stats([json.load(open(BK/f'control/pred-fem-{w}.json'))['seconds']/60 for w in FS.BW]))
# ---- stem (S1 push and S3 striker share the fits)
S = {}
S['GIC'] = dict(fit_initial=stats([gic_dir_minutes(ST/'gic/D0')]), refit_calibration=stats([gic_dir_minutes(ST/f'gic/D1-{w}') for w in FS.SW]),
                forecast=None)
_s = json.load(open(ST/'pac/D0/fit.json')).get('seconds')  # absent: that fit record was recovered from the checkpoint
S['PAC-NeRF'] = dict(fit_initial=stats([_s/60 if _s else None]),
                     refit_calibration=stats([json.load(open(ST/f'pac/D1-{w}/fit.json'))['seconds']/60 for w in FS.SW]),
                     forecast=stats([log_minutes(rf'pacnerf_stem.py forecast \S*pac/D1-{w} \S*forecast-D2c') for w in FS.SW]))
S['M-PhyGs'] = dict(fit_initial=stats([mphygs_fit_minutes('stem', 'D0', w, (12, 24)) for w in FS.SW]),
                    refit_calibration=stats([mphygs_fit_minutes('stem', 'D1', w, (12, 24, 36)) for w in FS.SW]),
                    forecast=stats([json.load(open(EXT/f'mphygs/forecast-D2c-D1-{w}.json'))['seconds']/60 for w in FS.SW]))
S['PhysTwin'] = dict(fit_initial=stats([json.load(open(EXT/f'phystwin/forecast-D2c-D0-{w}.json'))['fit']['wall_seconds']/60 for w in FS.SW]),
                     refit_calibration=stats([json.load(open(EXT/f'phystwin/forecast-D2c-D1-{w}.json'))['fit']['wall_seconds']/60 for w in FS.SW]),
                     forecast=stats([json.load(open(EXT/f'phystwin/forecast-D2c-D1-{w}.json'))['seconds']/60 for w in FS.SW]))
# ---- ours
t = json.load(open(OURS_TIMING)) if OURS_TIMING else {}
pick = lambda pre: [v/60 for k, v in t.items() if k.startswith(pre)]
bank_stem = (json.load(open(ST/'ours/bank.json'))['seconds']+json.load(open(ST/'ours/bank2.json'))['seconds'])/60
bank_block = json.load(open(BK/'ours/bank.json'))['seconds']/60
S['ours'] = dict(one_off_simulation_bank_cpu=stats([bank_stem]), tracking_per_episode=stats(pick('track stem')),
                 posterior_and_push_prediction=stats(pick('S1/S3 stem')),
                 striker_prediction=stats([json.load(open(ST/f'ours/striker/rgb-{w}-D0pD1.json'))['seconds']/60 for w in FS.SW]))
B['ours'] = dict(one_off_simulation_bank_cpu=stats([bank_block]), tracking_per_episode=stats(pick('track block')),
                 posterior_and_prediction=stats(pick('S2 block')))
out = dict(units='minutes (median over worlds, with min/max and n)', notes=__doc__, stem_S1_S3=S, block_S2=B)
(FS.REP/'compute-cost.json').write_text(json.dumps(out, indent=1))
for name, d in (('stem (S1, S3)', S), ('block (S2)', B)):
    print(f'== {name}')
    for m, st in d.items():
        print(f'  {m:12s} ' + ';  '.join(f"{k} {v['median']:.3g}" + (f" [{v['min']:.3g}-{v['max']:.3g}]" if v['n'] > 1 else '') for k, v in st.items() if v))
