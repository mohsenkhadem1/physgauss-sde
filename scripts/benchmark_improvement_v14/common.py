"""Shared campaign paths and the user-authorized absolute runtime bound."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/benchmark-improvement-v14'
REPORT = ROOT / 'reports/benchmark-improvement-v14'
CONFIG = ROOT / 'configs/benchmark_improvement_v14.json'
DEADLINE = datetime.fromisoformat('2026-10-31T18:36:34+00:00')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def remaining():
    if datetime.fromisoformat(read(CONFIG)['deadline_utc']) != DEADLINE:
        raise RuntimeError('Campaign deadline changed')
    seconds = (DEADLINE - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        raise RuntimeError('User-authorized twelve-hour campaign has ended')
    return seconds


def check_preservation():
    expected = read(REPORT / 'prior-preservation.json')['sha256']
    changed = [path for path, digest in expected.items()
               if not (ROOT / path).is_file() or sha(ROOT / path) != digest]
    if changed:
        raise RuntimeError('Historical artifacts changed: ' + repr(changed))
    return len(expected)
