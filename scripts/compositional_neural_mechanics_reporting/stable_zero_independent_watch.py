"""Read-only reporting waiter with original oracle deadline+600s."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';manifest=ROOT/'runs/compositional-neural-mechanics-v16/oracle-reserved-forward/run-manifest.json';deadline=json.loads(manifest.read_text())['deadline_unix']+600
while time.time()<deadline:
 audit=R/'oracle-final-independent-audit.json'
 if audit.exists()and json.loads(audit.read_text()).get('passed')and(R/'stable-zero-control-results.json').exists():
  result=subprocess.run([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.stable_zero_independent_audit'],cwd=ROOT,timeout=max(.1,min(180,deadline-time.time())));raise SystemExit(result.returncode)
 time.sleep(min(60,max(0,deadline-time.time())))
raise SystemExit('No complete independently audited parent and stable control scores before reporting deadline')
