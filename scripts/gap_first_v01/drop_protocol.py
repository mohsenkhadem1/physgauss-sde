"""Common immutable drop-campaign evidence verification."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/gap-first-v01/drop'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 p=REPORT/'protocol.json';d=json.loads(p.read_text())
 for f,h in d['frozen_sha256'].items():
  if sha(ROOT/f)!=h:raise RuntimeError('Frozen evidence mismatch: '+f)
 return d
