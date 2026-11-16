"""Write runs/pokeflex-v02/pred/MANIFEST.sha256.json (sha256 of every v02 prediction file, the calibration selection
record and every file in scripts/pokeflex_v02/) and copy it to reports/pokeflex-v02/MANIFEST.sha256.json.
Must run before any v02 held-out scoring; the copy is committed before scoring.  usage: python -I v02_manifest.py"""
import hashlib, json, shutil, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v02_model import ROOT, OUT2, PRED2, OBJECTS, heldout_pokes

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REP2 = ROOT/'reports/pokeflex-v02'


def main():
    man = PRED2/'MANIFEST.sha256.json'
    preds = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(PRED2.rglob('*')) if p.is_file() and p != man}
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    sel = OUT2/'selection.json'
    cover = {}
    for rung in sorted(d.name for d in PRED2.iterdir() if d.is_dir()):
        cover[rung] = {o: dict(heldout=len(heldout_pokes(o)), predicted=sum((PRED2/rung/o/f'{t}-p{k}.npz').exists() for t, k in heldout_pokes(o)))
                       for o in OBJECTS}
    rec = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               note='PokeFlex v02 mechanism ladder (post-hoc, non-blind; protocol scripts/pokeflex_v02/protocol_v02.json). '
                    'Recorded before any v02 held-out scoring.',
               prediction_files=preds, code_files=code, selection_record={sel.relative_to(ROOT).as_posix(): sha(sel)},
               selection=json.loads(sel.read_text())['selected'], coverage=cover)
    man.write_text(json.dumps(rec, indent=1))
    REP2.mkdir(parents=True, exist_ok=True); shutil.copyfile(man, REP2/'MANIFEST.sha256.json')
    print('manifest', len(preds), 'prediction files', len(code), 'code files; selected', rec['selection'])
    print(json.dumps({r: {o: f"{c['predicted']}/{c['heldout']}" for o, c in d.items()} for r, d in cover.items()}))


if __name__ == '__main__':
    main()
