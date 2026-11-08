"""Recompute original released analysis into memory and compare JSON scalar repair."""
import json
from pathlib import Path
from scripts.compositional_dynamics_v17 import cohort_reference_sensitivity as original
from scripts.compositional_dynamics_v17.cohort_reference_sensitivity_json import normalize
ROOT=original.ROOT;REPORT=original.REPORT

def main():
 protocol=REPORT/'fresh-cohort-protocol.json';freeze=REPORT/'fresh-cohort-method-freeze.json';release=REPORT/'fresh-cohort-reference-release.json';result=REPORT/'fresh-cohort-reference-sensitivity.json'
 bindings=dict(protocol_path=str(protocol),freeze_path=str(freeze),approval_path=str(REPORT/'fresh-cohort-prefit-independent-audit.json'),protocol_sha=original.sha(protocol),freeze_sha=original.sha(freeze))
 captured={};oldwriter=original.write;oldreport=original.REPORT
 class Routes:
  def __truediv__(self,name):
   if name=='fresh-cohort-reference-sensitivity.json':return REPORT/'serialization-check-output-NOT-WRITTEN.json'
   return REPORT/name
 def capture(path,value):captured.update(normalize(value))
 original.REPORT=Routes();original.write=capture
 try:original.main(bindings,str(release))
 finally:original.REPORT=oldreport;original.write=oldwriter
 actual=json.loads(result.read_text());note=actual.pop('serialization_note');wrapper='scripts/compositional_dynamics_v17/cohort_reference_sensitivity_json.py';wrapper_hash=actual['source_sha256'].pop(wrapper)
 if captured!=actual:raise ValueError('Scalar conversion changed computed scientific values')
 def count(x):return sum(count(v)for v in x.values())if isinstance(x,dict)else sum(count(v)for v in x)if isinstance(x,list)else 1
 report=dict(passed=True,all_computed_values_exactly_equal=True,scalar_leaf_count=count(captured),normalization='np.generic.item(), recursively through dict/list/tuple; no metric/gate/selection changes',first_failures=['Relative release path rejected by Path.relative_to before reference reads. Retried with absolute path.','Absolute-path original computation completed but JSON serialization rejected NumPy Boolean scalar; no output report was written.'],failure_record_scope='This log reconstructs the exact exceptions from tool transcripts; no claim a separate shell stderr file existed.',evidence_sha256={str(p.relative_to(ROOT)):original.sha(p)for p in [protocol,freeze,release,result,Path(original.__file__),ROOT/wrapper,Path(__file__)]},source_wrapper_hash=wrapper_hash,output_scope='Original released main recomputed into memory only; scalar-repaired report differs only by wrapper source hash and explicit serialization note.')
 original.write(REPORT/'reference-sensitivity-serialization-repair-audit.json',report)
 print(json.dumps(dict(passed=True,scalar_leaf_count=report['scalar_leaf_count'])))
if __name__=='__main__':main()
