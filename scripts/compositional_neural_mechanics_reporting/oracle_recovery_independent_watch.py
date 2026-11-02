import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/oracle-missing-recovery';deadline=json.loads((B/'start.json').read_text())['deadline_unix']+300
while time.time()<deadline:
 if (R/'oracle-recovery-merged-results.json').exists():
  r=subprocess.run([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.oracle_recovery_independent_audit'],cwd=ROOT,timeout=min(180,max(.1,deadline-time.time())));raise SystemExit(r.returncode)
 time.sleep(min(60,max(0,deadline-time.time())))
raise SystemExit('Recovery audit deadline; no restart')
