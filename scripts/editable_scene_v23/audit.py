"""Check pilot artifacts without turning numerical checks into physical validation."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/editable-scene-v23'

def main():
    checks=[]
    folders=list(BASE.glob('*/result.json'));assert len(folders)==10
    for path in folders:
        d=json.loads(path.read_text());assert d['complete'] and d['min_J']>0
        for p,h in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
        with np.load(path.parent/'states.npz')as data:
            assert np.isfinite(data['positions']).all() and np.isfinite(data['velocities']).all()
        if 'force_impulse_error_Ns'in d:assert d['force_impulse_error_Ns']<1e-7
        checks.append(path.parent.name)
    subprocess.run([str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'),'-v','error','-xerror','-i',str(BASE/'editable-scene-pilot.mp4'),'-f','null','-'],check=True,stderr=subprocess.PIPE)
    checks.append('video decoded')
    report=dict(passed=True,checks=checks,scope='Source integrity, finite states, positive sampled Jacobians, applied impulse and video decoding. Not calibrated uncertainty or real-world fidelity.')
    (ROOT/'reports/editable-scene-v23/audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

if __name__=='__main__':main()
