"""Read immutable protocol/method commitments, without any truth access."""
import json
from pathlib import Path
from scripts.physical_refinement_v15.measurement_broker import digest
ROOT=Path(__file__).resolve().parents[2]


def verify(protocol,protocol_sha,freeze,freeze_sha,*,artifacts=False):
    if digest(protocol)!=protocol_sha or digest(freeze)!=freeze_sha:raise ValueError('Commitment hash changed')
    p=json.loads(Path(protocol).read_text());f=json.loads(Path(freeze).read_text())
    if f['protocol_sha256']!=protocol_sha:raise ValueError('Protocol does not match method freeze')
    for key in ('source_sha256','artifact_sha256') if artifacts else ('source_sha256',):
        for name,expected in f[key].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or digest(path)!=expected:raise ValueError('Frozen input changed: '+name)
    return p,f
