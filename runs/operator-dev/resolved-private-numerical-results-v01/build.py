"""POSTHOC compact scalar extraction; does not run inference or access raw truth."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
source=ROOT/'runs/gaussian-campaign-v01/resolved-private-numerical-evaluation-v01/summary.json'
s=json.loads(source.read_text())
def rec(p):
 b=p.read_bytes();return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
compact=dict(schema='RESOLVED-numerical-development-compact-v01',completed=s['completed'],summary=rec(source),full_hash_and_PID_barrier=s['full_hash_and_PID_barrier'],SIDE_DOWN_observation_arrays_read=False,all64_denominators_retained=True,model_or_interval_changes=False,new_population_test=False,truth_previously_analyzed=True,models={},oracle_fixed4={})
for kind,models in s['models'].items():
 compact['models'][kind]={}
 for name,v in models.items():
  c=v['coverage'];d=dict(denominator_objects=64,denominator_coordinates_including_failures=1728,denominator_z_including_failures=576,numerically_admitted=c['numerically_admitted'],available_descriptive_predictions=c['available_descriptive_predictions'],pooled_available_coordinate_RMSE_um=v['metrics']['pooled_available_coordinate_RMSE_m']*1e6,material95_admitted_covered=sum(g['material95_covered_count']for g in v['metrics']['groups']),numerical_diagnostics=v['numerical_diagnostics'])
  for key in ('raw95_admitted','numerical_enclosure95_admitted','raw95_available_descriptive','numerical_enclosure95_available_descriptive'):
   a=c[key];d[key]=dict(covered=a['covered'],**{axis+'_covered':a['covered_by_axis'][i]for i,axis in enumerate(('x','y','z'))})
  compact['models'][kind][name]=d
for key,o in s['public_model_oracle_fixed4'].items():
 d={}
 for field in ('all63_displacement','selected9_displacement','all63_position','selected9_position'):
  a=o[field];d[field]=dict(coordinate_RMS_m=a['coordinate_RMS_m'],maximum_point_norm_m=a['maximum_point_norm_m'],**{axis+'_coordinate_RMS_m':a['coordinate_RMS_by_axis_m'][i]for i,axis in enumerate(('x','y','z'))},**{axis+'_maximum_absolute_m':a['maximum_absolute_by_axis_m'][i]for i,axis in enumerate(('x','y','z'))})
 d['position_vs_displacement_error_roundtrip_max_m']=o['position_vs_displacement_error_roundtrip_max_m'];compact['oracle_fixed4'][key]=d
save(OUT/'compact-result.json',compact)
receipt=ROOT/'runs/operator-dev/resolved-private-numerical-evaluation-v01/actual-execution-receipt.json';r=json.loads(receipt.read_text())
assert not Path('/proc',str(r['supervisor_pid'])).exists() and not Path('/proc',str(r['worker_pid'])).exists()
save(OUT/'completion-check.json',dict(completed=True,supervisor_PID_absent=True,worker_PID_absent=True,execution_receipt=rec(receipt),all87_frozen_source_pins_unchanged_at_evaluation_completion=True,raw_summary=rec(source),raw_per_record_outputs_local=True))
