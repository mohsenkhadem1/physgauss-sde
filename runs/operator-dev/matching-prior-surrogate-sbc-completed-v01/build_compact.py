"""Posthoc scalar-only summary of the COMPLETE SAME64 SBC scores; raw stays local."""
from pathlib import Path
import hashlib,json
from experiments import run_campaign_surrogate_sbc_evaluation_continuation as c
ROOT=Path(__file__).resolve().parents[3];OP=Path(__file__).resolve().parent
PROTOCOL='runs/operator-dev/matching-prior-surrogate-sbc-evaluation-continuation-v01/protocol.json';DIGEST='6380cd738fe26a4212690648cde7ea812f3c6e86193dbebb5d3dece5cae6fb65'

def rec(p):
 b=p.read_bytes();return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def save(name,v):c.io.write_new(OP/name,v)

p=c.validate(PROTOCOL,DIGEST,root=ROOT);out=c.io.local(ROOT,p['output']);execution=c.io.read_json(ROOT,c.io.record(ROOT,out/'execution.json'));seal=c.io.read_json(ROOT,c.io.record(ROOT,out/'merged-score-seal.json'));summary=c.io.read_json(ROOT,c.io.record(ROOT,out/'summary.json'));supfile=ROOT/'runs/operator-dev/matching-prior-surrogate-sbc-evaluation-continuation-v01/actual-supervisor-receipt.json';sup=json.loads(supfile.read_text())
assert execution['completed']and execution['all_owned_workers_reaped']and len(execution['records'])==55 and sup['completed']and sup['controller_pid']==execution['pid']
pids=[execution['pid'],sup['supervisor_pid']]+[r['pid']for r in execution['records']]
assert len(set(pids))==57 and all(not Path('/proc',str(pid)).exists()for pid in pids)
for stage in ('generate','inference','evaluation'):
 prior=json.loads((ROOT/('runs/operator-dev/matching-prior-surrogate-sbc-v02/actual-'+stage+'-receipt.json')).read_text())
 assert all(not Path('/proc',str(pid)).exists()for pid in [prior['supervisor_pid'],prior['controller_pid']]+prior['owned_worker_PIDs_observed'])
rows,values=c.merged_rows(ROOT,p,execution['records']);assert seal['completed']and seal['count']==64 and seal['objects']==rows and summary['completed']and summary['objects']==64
for r in p['inputs']['reused']:
 assert next(z for z in seal['objects']if z['object_index']==r['object_index'])==r
# Recompute scalar aggregate from all64 retained rows, never filter numerics.
for phase in c.sbc.PHASES:assert summary['phases'][phase]==c.sbc.aggregate([v['phases'][phase]for v in values])
compact=dict(schema='MATCHING-PRIOR-NL-SURROGATE-SBC-complete-compact-v01',completed=True,independent_objects=64,phases_paired=True,phase_laws=128,reused_complete_scores=9,continued_scores=55,all64_denominators_retained=True,all88_source_pins_and_original_seals_verified=True,all_continuation57_owned_PIDs_absent=True,original_generation_seal=p['inputs']['generation_seal'],original_inference_seal=p['inputs']['inference_seal'],preserved_technical_stop=p['inputs']['original_technical_stop'],original_partial_journal=p['inputs']['original_partial_journal'],continuation_protocol=dict(path=PROTOCOL,sha256=DIGEST),merged_score_seal=rec(out/'merged-score-seal.json'),local_raw_summary=rec(out/'summary.json'),supervisor_receipt=rec(supfile),new_generation_or_inference=False,model_or_seed_changes=False,physical_population_calibration=False,active_policy_experiment=False,ordinary_atom_credible_interval_coverage=False,coverage_semantics='Randomized equal-tail PIT acceptance, per quantity across64 independent prior draws; coordinates correlated and phases paired; descriptive binomial uncertainty, no multiplicity-adjusted claim',phases={})
for phase,aggregate in summary['phases'].items():
 d=dict(denominator_objects=64,numerically_admitted=aggregate['numerically_admitted'],quantities={},methods={})
 for key,q in aggregate['quantities'].items():
  j=int(key);name='material_E'if j==0 else'q'+str(int(c.sbc.POINT_IDS[(j-1)//3]))+'_'+('xyz'[(j-1)%3]);new={k:v for k,v in q.items()if k!='PIT_histogram_counts'};new['histogram']={str(i):n for i,n in enumerate(q['PIT_histogram_counts'])};d['quantities'][name]=new
 for method in c.sbc.METHODS:
  ms=[v['phases'][phase]['evaluation']['methods'][method]for v in values];available=[r for r in ms if r['completed']];all_scores=[s for r in available for s in r['scores']];d['methods'][method]=dict(completed_objects=len(available),numerical_failed_objects=64-len(available),available_PIT_quantities=len(all_scores),expected_PIT_quantities=64*28,atom_PIT_count=sum(s['atom']for s in all_scores),numerically_unqualified_PIT_count=sum(not s['numerically_qualified']for s in all_scores),maximum_PIT_numerical_width=max((s['numerical_width']for s in all_scores),default=None))
 differences=[v['phases'][phase]['evaluation']['maximum_method_PIT_difference']for v in values if v['phases'][phase]['evaluation']['maximum_method_PIT_difference']is not None];d['maximum_method_PIT_difference']=max(differences,default=None);compact['phases'][phase]=d
save('compact-result.json',compact)
save('completion-receipt.json',dict(completed=True,all57_continuation_PIDs_absent=True,all_original_generation_inference_partial_evaluation_PIDs_absent=True,all88_continuation_source_pins_verified=True,all9_reused_bundle_file_hashes_and_score_values_identical=True,all55_remaining_scores_completed=True,all64_aggregate_recomputed_identically=True,original_technical_stop_preserved=True,raw_private_rows_local=True))
print(json.dumps({k:dict(admitted=v['numerically_admitted'],material=v['quantities']['material_E'],maximum_method_PIT_difference=v['maximum_method_PIT_difference'],methods=v['methods'])for k,v in compact['phases'].items()}))
