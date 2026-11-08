"""Crash-resistant post-fit scheduler; no retries, no scientific source edits."""
from pathlib import Path
import json,subprocess,time,os
R=Path(__file__).resolve().parents[2];B=R/'runs/gap-crossmethod-v02';OUT=R/'reports/gap-crossmethod-v02';state=dict(pid=os.getpid(),phase='waiting_for_native_fit');p=OUT/'pacnerf-finalization.json'
def save():state['updated_unix']=time.time();p.write_text(json.dumps(state,indent=2))
while True:
 save();d=json.loads((OUT/'pacnerf-prefix-status.json').read_text())
 if d['phase']=='failed':state.update(phase='blocked_native_fit_failure',error=d.get('error'));save();raise SystemExit(1)
 if d['phase']=='complete':break
 if time.time()>d['started']+21600:state.update(phase='deadline_incomplete');save();raise SystemExit(1)
 time.sleep(30)
for script in ['pacnerf_baseline_metrics.py','pacnerf_counterfactual.py']:
 state.update(phase='running',script=script);save()
 with (B/(script+'.log')).open('w') as log:
  q=subprocess.run(['timeout','3600','python3','scripts/gap_crossmethod_v02/pacnerf_launch.py','scripts/gap_crossmethod_v02/'+script],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 if q.returncode:state.update(phase='postprocess_failed',returncode=q.returncode);save();raise SystemExit(q.returncode)
state.update(phase='complete');save()
