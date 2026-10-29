"""Stop only this campaign's systemd services at its absolute deadline."""
from pathlib import Path
import datetime,json,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
spec=json.loads((ROOT/'configs/position_ood_v09.json').read_text())
deadline=datetime.datetime.fromisoformat(spec['campaign_deadline_utc'].replace('Z','+00:00')).timestamp()
while time.time()<deadline:time.sleep(min(30,max(0,deadline-time.time())))
result=subprocess.check_output(['systemctl','--user','list-units','position-ood-v09-*','--state=running','--no-legend','--plain'],text=True)
units=[line.split()[0] for line in result.splitlines() if line.strip() and 'deadline' not in line.split()[0]]
if units:subprocess.run(['systemctl','--user','stop',*units],check=True)
(ROOT/'runs/position-ood-v09/deadline-reached.json').write_text(json.dumps(dict(time=time.time(),stopped_units=units),indent=2)+'\n')
