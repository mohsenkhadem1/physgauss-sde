"""Additive reference-state/cohort guard around the frozen control scorer."""
import json
from scripts.compositional_neural_mechanics import stable_zero_evaluate as evaluator
from scripts.compositional_neural_mechanics.stable_zero_control import guard,ROOT,REPORT,sha
from scripts.compositional_neural_mechanics.dataset_runner import write

def verify_references():
 guard();plan,records=evaluator.runtime_records();commit=json.loads(evaluator.COMMIT.read_text())
 expected={(r['scene']['name'],r['role'],a['name'],m)for r in records for a in r['actions']for m in guard()['variants']}
 if len(commit['rows'])!=96 or {(r['scene_id'],r['cohort'],r['action'],r['method'])for r in commit['rows']}!=expected:raise ValueError('Exact forecast cohort identity required')
 evidence={}
 for record in records:
  folder=evaluator.ORIGINAL/record['scene']['name'];status=folder/'complete.json'
  if not status.exists():raise ValueError('Parent reference scene must be terminal')
  rows=json.loads(status.read_text())['rows'];evidence[str(status.relative_to(ROOT))]=sha(status)
  for action in record['actions']:
   matches=[r for r in rows if r['method']=='reference'and r['action']==action['name']]
   if len(matches)!=1:raise ValueError('Unique parent reference required')
   row=matches[0];path=folder/action['name']/'reference.npz'
   if row['state']!='complete' or row['scene_id']!=record['scene']['name'] or row['cohort']!=record['role'] or ROOT/row['artifact']!=path or sha(path)!=row['sha256']:raise ValueError('Reference completion/identity/hash mismatch')
   evidence[str(path.relative_to(ROOT))]=sha(path)
 return evidence

def main():
 evidence=verify_references();evaluator.score()
 if evidence!=verify_references():raise ValueError('Reference evidence changed during scoring')
 write(REPORT/'stable-zero-scoring-reference-audit.json',dict(passed=True,reference_evidence_sha256=evidence,scorer_sha256=sha(__file__),forecast_commit_sha256=sha(evaluator.COMMIT)))
if __name__=='__main__':main()
