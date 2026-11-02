"""Bounded reporting-only waiter; no model changes or partial-reference scoring."""
import json,time
from pathlib import Path
from scripts.compositional_neural_mechanics import stable_zero_score_guarded as scorer
from scripts.compositional_neural_mechanics.stable_zero_control import ROOT,REPORT,OUT,PLAN,sha,guard
from scripts.compositional_neural_mechanics.dataset_runner import write

def main():
 guard();_,records=scorer.evaluator.runtime_records();parent=scorer.evaluator.ORIGINAL
 manifest=parent/'run-manifest.json';deadline=float(json.loads(manifest.read_text())['deadline_unix'])+600
 path=REPORT/'stable-zero-watcher-status.json'
 pins={str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__).resolve(),Path(scorer.__file__).resolve(),PLAN,manifest)}
 while time.time()<deadline:
  if any(sha(ROOT/p)!=h for p,h in pins.items()):raise ValueError('Watcher sources/plan changed')
  count=sum((parent/r['scene']['name']/'complete.json').exists()for r in records)
  write(path,dict(state='waiting',completed_scenes=count,expected_scenes=12,deadline_unix=deadline,evidence_sha256=pins))
  if count==12:
   try:
    scorer.main();state='scored';error=None
   except Exception as exc:state='blocked_reference_or_guard';error=repr(exc)
   write(path,dict(state=state,error=error,deadline_unix=deadline,evidence_sha256=pins,independent_audit_requested=True));return
  time.sleep(min(30,max(0,deadline-time.time())))
 write(path,dict(state='deadline_expired_no_scoring',deadline_unix=deadline,evidence_sha256=pins))
if __name__=='__main__':main()
