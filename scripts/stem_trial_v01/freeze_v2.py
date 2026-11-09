"""Seal round-2 (D2b) predictions and GIC appearance checkpoints before D2b truth exists."""
import hashlib, json, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not list((T/'truth').glob('D2b-*')), 'D2b truth already exists'
files = {}
for p in sorted((T/'gic').glob('forecast-D2b-*.npz')):
    files[str(p.relative_to(T))] = p
for d in ('ours', 'ours-offgrid'):
    for p in sorted((T/d).glob('round2-*')):
        files[str(p.relative_to(T))] = p
for p in sorted((T/'ours-fields').glob('*-D2b.npz')):
    files[str(p.relative_to(T))] = p
for m in ['D0', 'D1-W500', 'D1-W1000', 'D1-W2000']:
    for p in sorted((T/f'gic/{m}/point_cloud_fix_pcd').rglob('point_cloud.ply')):
        files[str(p.relative_to(T))] = p
expect = 7+18+6+4
assert len(files) >= expect, (len(files), sorted(files))
code = {p.name: sha(p) for p in sorted((R/'scripts/stem_trial_v01').glob('*.py'))}
code['protocol_v02.json'] = sha(R/'scripts/stem_trial_v01/protocol_v02.json')
rec = dict(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files={k: sha(p) for k, p in files.items()},
           code_sha256=code, rule='D2b truth generated only after this file exists')
out = R/'reports/stem-trial-v01/prediction-freeze-v2.json'
assert not out.exists()
out.write_text(json.dumps(rec, indent=1)); print(rec['sealed_utc'], len(files), 'files sealed')
