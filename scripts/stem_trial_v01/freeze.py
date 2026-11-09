"""Seal all D2 predictions before D2 truth exists. Refuses if any D2 truth file is present."""
import hashlib, json, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not list((T/'truth').glob('D2-*')), 'D2 truth already exists: cannot seal prospectively'
files = {}
for w in ['W500', 'W1000', 'W2000']:
    files[f'ours/pred-{w}.npz'] = T/f'ours/pred-{w}.npz'
    files[f'ours/pred-{w}.json'] = T/f'ours/pred-{w}.json'
    files[f'ours-offgrid/pred-{w}.npz'] = T/f'ours-offgrid/pred-{w}.npz'
    files[f'ours-offgrid/pred-{w}.json'] = T/f'ours-offgrid/pred-{w}.json'
    files[f'gic/forecast-D1-{w}.npz'] = T/f'gic/forecast-D1-{w}.npz'
    files[f'gic/D1-{w}/0-pred.json'] = T/f'gic/D1-{w}/0-pred.json'
    files[f'gic/forecast-oracle-{w}.npz'] = T/f'gic/forecast-oracle-{w}.npz'
files['gic/forecast-D0.npz'] = T/'gic/forecast-D0.npz'
files['gic/D0/0-pred.json'] = T/'gic/D0/0-pred.json'
missing = [k for k, p in files.items() if not p.exists()]
code = {p.name: sha(p) for p in sorted((R/'scripts/stem_trial_v01').glob('*.py'))}
code['protocol.json'] = sha(R/'scripts/stem_trial_v01/protocol.json')
rec = dict(sealed_unix=time.time(), sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           predictions={k: sha(p) for k, p in files.items() if p.exists()}, missing_predictions=missing,
           code_sha256=code, rule='D2 truth generated only after this file exists; scoring verifies these hashes')
out = R/'reports/stem-trial-v01/prediction-freeze.json'
assert not out.exists(), 'freeze already written'
out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
