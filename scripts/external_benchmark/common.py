"""Isolated external-data campaign utilities. Raw data must remain under runs/."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/external-benchmark-v11'
REPORT = ROOT / 'reports/external-benchmark-v11'
CONFIG = ROOT / 'configs/external_benchmark_v11.json'
_registered = False

def check_deadline():
    global _registered
    if not _registered:
        proc = Path('/proc/self/stat').read_text().split(') ',1)[1].split()
        write_json(BASE/'pids'/f'{os.getpid()}.json',dict(pid=os.getpid(),start_ticks=proc[19]))
        _registered = True
    config = json.loads(CONFIG.read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(config['deadline_utc'].replace('Z', '+00:00')):
        raise TimeoutError('External benchmark campaign reached its 12-hour deadline')

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def save_provenance(directory, data):
    import sys
    sources={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts/external_benchmark').glob('*.py')}
    write_json(Path(directory)/'provenance.json',dict(
        created_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        source_files=sources,input_file=data['source'],input_sha256=sha256(data['source']),
        python=sys.version,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        prediction_inputs='initial geometry, recorded tool positions, observed prefix geometry only'))
