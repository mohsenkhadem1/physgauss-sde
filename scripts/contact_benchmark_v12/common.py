"""Isolated v12 campaign paths and hard deadline; v11 evidence is read-only."""
from pathlib import Path
from datetime import datetime, timezone
import os, json
from scripts.external_benchmark.common import write_json, sha256
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/contact-benchmark-v12'
REPORT = ROOT / 'reports/contact-benchmark-v12'
CONFIG = ROOT / 'configs/contact_benchmark_v12.json'

def check_deadline():
    spec=json.loads(CONFIG.read_text())
    if datetime.now(timezone.utc)>=datetime.fromisoformat(spec['deadline_utc'].replace('Z','+00:00')):
        raise TimeoutError('Contact campaign reached its 12-hour maximum')
    record=BASE/'pids'/f'{os.getpid()}.json'
    if not record.exists():
        ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]
        write_json(record,dict(pid=os.getpid(),start_ticks=ticks))

def provenance(out, data):
    write_json(Path(out)/'provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_file=data['source'],input_sha256=sha256(data['source']),
        source_files={str(p.relative_to(ROOT)):sha256(p) for folder in ['scripts/contact_benchmark_v12','scripts/external_benchmark'] for p in (ROOT/folder).glob('*.py')},
        raw_data_local_only=True))
