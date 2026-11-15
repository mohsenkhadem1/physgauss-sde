"""Write runs/pokeflex-v01/pred-stageB/MANIFEST.sha256.json: sha256 of every stage-B prediction file and of every file
in scripts/pokeflex_stageb/ (record made before the v01 freeze; stage B is reported separately, never mixed in).
Also lists the held-out pokes per method/object and which are missing.  usage: python -I manifest.py"""
import hashlib, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sb_common import R, PRED_B, OBJECTS, obj_info

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    man = PRED_B/'MANIFEST.sha256.json'
    preds = {p.relative_to(R).as_posix(): sha(p) for p in sorted(PRED_B.rglob('*')) if p.is_file() and p != man}
    code = {p.relative_to(R).as_posix(): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    cover = {}
    for meth in ('PhysTwin', 'PAC-NeRF'):
        cover[meth] = {}
        for obj in OBJECTS:
            ho = obj_info(obj)['heldout']
            have = [f'{t}-p{k}' for t, k in ho if (PRED_B/meth/obj/f'{t}-p{k}.npz').exists()]
            cover[meth][obj] = dict(heldout=len(ho), predicted=len(have),
                                    missing=[f'{t}-p{k}' for t, k in ho if f'{t}-p{k}' not in have])
    rec = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), note='stage B (PhysTwin, PAC-NeRF) blind predictions; '
               'not part of the v01 freeze', prediction_files=preds, code_files=code, coverage=cover)
    man.write_text(json.dumps(rec, indent=1))
    print('manifest', len(preds), 'prediction files', len(code), 'code files')
    print(json.dumps({m: {o: f"{c['predicted']}/{c['heldout']}" for o, c in d.items()} for m, d in cover.items()}, indent=1))


if __name__ == '__main__':
    main()
