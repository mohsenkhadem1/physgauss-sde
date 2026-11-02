"""Reporting-only audit-gated mixed-reference scoring and independent verification."""
import json,subprocess,sys,time
from scripts.compositional_neural_mechanics_reporting import oracle_missing_recovery as recovery
from scripts.compositional_neural_mechanics.dataset_runner import write
R=recovery.R;ROOT=recovery.ROOT;sha=recovery.sha;PLAN=R/'stable-zero-recovery-reporting-amendment.json'
def main():
 p=json.loads(PLAN.read_text());deadline=p['deadline_unix'];status=R/'stable-zero-recovery-waiter-status.json'
 while time.time()<deadline:
  recovery.guard()
  for path,h in p['source_sha256'].items():
   if sha(ROOT/path)!=h:raise ValueError('Reporting source changed')
  audit=R/'oracle-recovery-final-independent-audit.json'
  if audit.exists():
   a=json.loads(audit.read_text())
   if not a.get('passed')or a.get('errors'):raise ValueError('Merged recovery audit failed')
   for path,h in a['evidence_sha256'].items():
    if sha(ROOT/path)!=h:raise ValueError('Merged audit stale')
   for module in ('stable_zero_mixed_references','stable_zero_mixed_independent_audit'):
    subprocess.run([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.'+module],check=True,timeout=min(300,max(1,deadline-time.time())))
   write(status,dict(state='scored_and_independently_audited',amendment_sha256=sha(PLAN)));return
  write(status,dict(state='waiting_merged_independent_audit',deadline_unix=deadline,amendment_sha256=sha(PLAN)));time.sleep(min(30,max(0,deadline-time.time())))
 write(status,dict(state='deadline_without_scoring',amendment_sha256=sha(PLAN)))
if __name__=='__main__':main()
