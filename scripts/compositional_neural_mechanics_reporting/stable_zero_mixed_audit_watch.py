"""Bounded reporting sequence only; never regenerates predictions/references."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';PLAN=R/'stable-zero-mixed-reporting-plan.json';p=json.loads(PLAN.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();deadline=p['deadline_unix']
while time.time()<deadline:
 for path,h in p['source_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('Reporting source changed')
 audit=R/'oracle-recovery-final-independent-audit.json'
 if audit.exists()and json.loads(audit.read_text()).get('passed'):
  stages=[]
  if not (R/'stable-zero-mixed-reference-results.json').exists():stages.append('scripts.compositional_neural_mechanics_reporting.stable_zero_mixed_references')
  stages.append('scripts.compositional_neural_mechanics_reporting.stable_zero_mixed_independent_audit')
  for stage in stages:
   result=subprocess.run([sys.executable,'-m',stage],cwd=ROOT,timeout=max(.1,min(240,deadline-time.time())))
   if result.returncode:raise SystemExit(result.returncode)
  raise SystemExit(0)
 time.sleep(min(60,max(0,deadline-time.time())))
raise SystemExit('Reporting deadline without recovered oracle audit; no restart')
