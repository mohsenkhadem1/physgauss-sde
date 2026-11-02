"""Score already committed stable controls against audited mixed reference index."""
import json
import numpy as np
from scripts.compositional_neural_mechanics_reporting import oracle_missing_recovery as recovery
from scripts.compositional_neural_mechanics_reporting.stable_zero_io_recovery import implementation, PLAN as IO_PLAN
control=implementation()
from scripts.compositional_neural_mechanics.stable_zero_control import guard as control_guard
from scripts.compositional_neural_mechanics.forward_evaluation import score_prediction
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.dataset_runner import write
ROOT=recovery.ROOT;R=recovery.R;sha=recovery.sha
AUDIT=R/'oracle-recovery-final-independent-audit.json'

def main():
 recovery.guard();control_guard()
 io_plan=json.loads(IO_PLAN.read_text())
 for path,h in io_plan['source_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('I/O replay source changed')
 prediction_audit=json.loads((R/'stable-zero-io-recovery-prediction-audit.json').read_text())
 if not prediction_audit['passed']or prediction_audit['successful']!=96 or prediction_audit['commit_sha256']!=sha(control.COMMIT)or prediction_audit['plan_sha256']!=sha(IO_PLAN):raise ValueError('Successful96replay audit required')
 audit=json.loads(AUDIT.read_text())
 if not audit.get('passed')or audit.get('errors'):raise ValueError('Passed merged recovery audit required')
 for path,h in audit['evidence_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('Recovery audit stale')
 indexpath=R/'oracle-recovery-reference-index.json';mergedpath=R/'oracle-recovery-merged-results.json'
 if any(str(p.relative_to(ROOT))not in audit['evidence_sha256']for p in (indexpath,mergedpath)):raise ValueError('Audit mustbindmergedledger/referenceindex')
 index=json.loads(indexpath.read_text());merged=json.loads(mergedpath.read_text());commit=json.loads(control.COMMIT.read_text());plan,records=control.runtime_records();variants=control_guard()['variants']
 if index['plan_sha256']!=sha(recovery.PLAN)or merged['reference_index_sha256']!=sha(indexpath):raise ValueError('Referenceindex binding changed')
 if commit['control_plan_sha256']!=sha(control.PLAN)or commit['parent_freeze_sha256']!=sha(control.FREEZE)or commit['records_sha256']!=sha(control.ORIGINAL/'reserved-records.json'):raise ValueError('Stable forecast commitment changed')
 expected={(r['scene']['name'],r['role'],a['name'],m)for r in records for a in r['actions']for m in variants};rows=commit['rows']
 if len(rows)!=96 or {(r['scene_id'],r['cohort'],r['action'],r['method'])for r in rows}!=expected:raise ValueError('Exact96control forecast rows required')
 references=index['rows'];expectedrefs={(r['scene']['name'],r['role'],a['name'])for r in records for a in r['actions']}
 if len(references)!=48 or {(r['scene_id'],r['cohort'],r['action'])for r in references}!=expectedrefs:raise ValueError('Exact48reference identity required')
 evidence={str(p.relative_to(ROOT)):sha(p)for p in (AUDIT,indexpath,mergedpath,control.COMMIT,recovery.PLAN,control.PLAN,IO_PLAN,R/'stable-zero-io-recovery-prediction-audit.json')}
 for row in rows:
  if row['state']not in ('complete','failed'):raise ValueError('Terminal control state required')
  if row['state']!='complete':continue
  if sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Control prediction changed')
  evidence[row['artifact']]=row['sha256'];ref=next(x for x in references if x['scene_id']==row['scene_id']and x['action']==row['action'])
  if ref['state']!='complete':row['scoring_error']='Reference failed original and bounded recovery';continue
  if sha(ROOT/ref['artifact'])!=ref['sha256']:raise ValueError('Mixed reference changed')
  evidence[ref['artifact']]=ref['sha256'];scene=next(r['scene']for r in records if r['scene']['name']==row['scene_id'])
  with np.load(ROOT/row['artifact'],allow_pickle=False)as prediction,np.load(ROOT/ref['artifact'],allow_pickle=False)as reference:
   for data in (prediction,reference):
    recovery.original.validate_value(data,plan['material_query'])
    if not np.array_equal(data['material_query'],np.asarray(plan['material_query'])):raise ValueError('Sharedquerycoordinates required')
   row['scores']=score_prediction(prediction,reference,GeometrySpec(**scene['geometry']).map(np.asarray(plan['material_query'])))
 summaries=[]
 for cohort in ('reserved_composition','reserved_geometry'):
  for method in variants:
   selected=[r for r in rows if r['cohort']==cohort and r['method']==method];valid=[r for r in selected if'scores'in r];entry=dict(cohort=cohort,method=method,expected_commands=24,scored_commands=len(valid),complete=len(valid)==24)
   if len(valid)==24:entry['object_mean_scores']={k:float(np.mean([r['scores'][k]for r in valid]))for k in valid[0]['scores']}
   summaries.append(entry)
 write(R/'stable-zero-io-recovered-mixed-results.json',dict(scope='Directory-onlyreplay of96frozenstablecontrols; original96forecastwritesfailed. SuccessfulartifactsproducedafterIOfixandvisibleprimaryoutcomes; no newtraining/settings/modelselection. Mixedoriginal/recoveredreferencepaths afterindependentaudit. Original5methodfailureledger preserved.',evidence_sha256=evidence,rows=rows,summaries=summaries,scorer_sha256=sha(__file__),runtime_disclosure='Existing stable predictions were generated onCPU inphystwin-env torch2.4.0+cu121; original/recovered oracle direct predictions use.venv torch2.14.1+cu130 onGPU. Controls retained unchanged; no crossruntime bitidentity claim.'))
 combined=[dict(r,source='bounded_missing_call_recovery')for r in merged['summaries']]+[dict(r,source='supplementary_stable_zero_control')for r in summaries]
 write(R/'oracle-recovered-plus-stable-io-comparison.json',dict(scope='Knownoraclephysicaldescriptors. Allfiveunchangedmethods andbothstablecontrols retained. Originalledger has48missingrows; recoveredledger reuses240andonlycomputesmissing48.',original_failure_ledger_sha256=sha(recovery.ORIGINAL_RESULT),merged_ledger_sha256=sha(mergedpath),stable_results_sha256=sha(R/'stable-zero-io-recovered-mixed-results.json'),rows=combined))
if __name__=='__main__':main()
