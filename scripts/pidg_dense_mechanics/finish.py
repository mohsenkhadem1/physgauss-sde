"""Durable completion pipeline; never silently publish an incomplete scene set."""
from pathlib import Path
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/pidg-dense-mechanics-v01'
SCENES = ['balls_reaction', 'rubber_duck', 'motion_kuro']
PY = Path.home() / '.local/share/physgauss-external/pidg-official-v01/env/bin/python'
VIS = ROOT / 'runs/phystwin-env/bin/python'
while not all((REPORT / s / 'results.json').exists() for s in SCENES):
    time.sleep(20)
for executable, script in [('python3', 'report.py'), (PY, 'audit_campaign.py'), (VIS, 'figures.py'), (VIS, 'videos.py')]:
    print('Running', script, flush=True)
    subprocess.run([str(executable), str(ROOT / 'scripts/pidg_dense_mechanics' / script)], cwd=ROOT, check=True)
summary = json.loads((REPORT / 'summary.json').read_text())
(REPORT / 'STATUS.txt').write_text(f"Complete: three scenes; {summary['primary_scenes_meeting_parity']}/3 meet predeclared rendering parity at width 320. See REPORT.txt and campaign-audit.json.\n")
print('All scene results, audits and visual artifacts complete.', flush=True)
