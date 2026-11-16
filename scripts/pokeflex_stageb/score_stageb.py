"""Score stage-B (PhysTwin, PAC-NeRF) PokeFlex predictions with the unchanged v01 scorer.

Blindness: every scored file must match the sha256 recorded in reports/pokeflex-v01/stageB-MANIFEST.sha256.json, and
that manifest must be byte-identical to its committed version (commit 5277bc1d, made before the v01 freeze).  The v01
freeze must also verify (it opens the held-out gate).  Scoring code is pfx_score.score_poke, unmodified; only its
prediction root and its 'sealed' check are redirected to the stage-B manifest.
usage: python -I score_stageb.py   -> reports/pokeflex-v01/scores-stageB.json"""
import hashlib, json, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/pokeflex_v01'))
import pokeflex as P
import pfx_score as S
from pfx_common import OBJECTS, heldout_pokes

MAN = ROOT/'reports/pokeflex-v01/stageB-MANIFEST.sha256.json'
MAN_COMMIT = '5277bc1d'
SB = ROOT/'runs/pokeflex-v01/pred-stageB'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_manifest():
    committed = subprocess.run(['git', '-C', str(ROOT), 'show', f'{MAN_COMMIT}:{MAN.relative_to(ROOT)}'],
                               capture_output=True, check=True).stdout
    if committed != MAN.read_bytes():
        raise SystemExit('stage-B manifest differs from its committed version')
    m = json.loads(committed)
    bad = [f for f, h in m['prediction_files'].items() if not (ROOT/f).exists() or sha(ROOT/f) != h]
    if bad:
        raise SystemExit(f'stage-B predictions changed since manifest: {bad[:5]}')
    return m


MANIFEST = load_manifest()
SEALED = {str((ROOT/f).resolve()) for f in MANIFEST['prediction_files']}
P.is_prediction_sealed = lambda f: str(Path(f).resolve()) in SEALED
S.PRED = SB


def main():
    fz = P.verify()
    methods = sorted(MANIFEST['coverage'])
    jobs = [(m, o, t, k) for m in methods for o in OBJECTS for t, k in heldout_pokes(o)]
    with Pool(24) as pool:
        rows = [r for r in pool.map(S.score_poke, jobs, chunksize=1) if r]
    agg = {}
    for m in methods:
        for o in OBJECTS:
            rr = [r for r in rows if r['method'] == m and r['object'] == o]
            if rr:
                agg.setdefault(m, {})[o] = {k: float(np.mean([r[k] for r in rr if k in r])) for k in rr[0]
                                            if isinstance(rr[0][k], float)} | dict(pokes=len(rr))
    out = dict(v01_freeze_utc=fz['sealed_utc'], manifest_created_utc=MANIFEST['created_utc'], manifest_commit=MAN_COMMIT,
               surface_every=S.EVERY, per_object=agg, rows=rows)
    (P.REP/'scores-stageB.json').write_text(json.dumps(out, indent=1))
    for m, d in agg.items():
        print(m, {o: round(v['force_rmse_N'], 2) for o, v in d.items()})


if __name__ == '__main__':
    main()
