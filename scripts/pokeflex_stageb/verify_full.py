"""Full verification of the PokeFlex v01 freeze (closes a gap in pokeflex.verify()).

pokeflex.freeze() records code_sha256 (every scripts/pokeflex_v01/*.py) and objects_sha256, but pokeflex.verify()
re-checks only prediction files, protocol.json and the sealed data manifests.  pokeflex.py itself cannot be changed
after the freeze without breaking its own recorded hash, so the missing checks live here.  This checks:
  1. pokeflex.verify()  (prediction files, protocol.json, sealed data manifests);
  2. every recorded code hash against the current file, and that no unrecorded .py file has appeared;
  3. objects.json against objects_sha256;
  4. the same code/protocol/objects hashes against the pre-freeze commit (PRE_FREEZE_COMMIT), i.e. the code that made
     the predictions is the code that was committed before the gate opened;
  5. the freeze record itself is byte-identical to its committed version (FREEZE_COMMIT).
usage: python -I verify_full.py   (exit 0 and 'ok' on success; raises otherwise).  Imported by compare.py."""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V01 = ROOT/'scripts/pokeflex_v01'
sys.path.insert(0, str(V01))
import pokeflex as P

PRE_FREEZE_COMMIT = 'fa1a5c9c'
FREEZE_COMMIT = '01fbd190'
sha = lambda b: hashlib.sha256(b).hexdigest()


def git_blob(commit, rel):
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'], capture_output=True, check=True).stdout


def verify_full():
    fz = P.verify()
    bad = []
    rec = fz['code_sha256']
    now = {p.name for p in V01.glob('*.py')}
    bad += [f'unrecorded code file {n}' for n in sorted(now-set(rec))]
    for name, h in rec.items():
        f = V01/name
        if not f.exists() or sha(f.read_bytes()) != h:
            bad.append(f'code changed since freeze: {name}')
        if sha(git_blob(PRE_FREEZE_COMMIT, f'scripts/pokeflex_v01/{name}')) != h:
            bad.append(f'code at freeze differs from pre-freeze commit {PRE_FREEZE_COMMIT}: {name}')
    if sha((V01/'objects.json').read_bytes()) != fz['objects_sha256']:
        bad.append('objects.json changed since freeze')
    for name, key in (('objects.json', 'objects_sha256'), ('protocol.json', 'protocol_sha256')):
        if sha(git_blob(PRE_FREEZE_COMMIT, f'scripts/pokeflex_v01/{name}')) != fz[key]:
            bad.append(f'{name} at freeze differs from pre-freeze commit')
    if git_blob(FREEZE_COMMIT, 'reports/pokeflex-v01/prediction-freeze.json') != P.FREEZE.read_bytes():
        bad.append(f'freeze record differs from committed version {FREEZE_COMMIT}')
    if bad:
        raise P.SealedError('; '.join(bad))
    return fz


if __name__ == '__main__':
    fz = verify_full()
    print('ok', len(fz['files']), 'prediction files,', len(fz['code_sha256']), 'code files, objects, protocol, freeze record')
