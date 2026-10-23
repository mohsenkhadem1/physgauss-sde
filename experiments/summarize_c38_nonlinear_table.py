"""Read-only terminal forty-state metadata seal; no trajectory array loading."""
import json
from pathlib import Path
from experiments import run_campaign_c38_nonlinear_table as execution
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/summarize_c38_nonlinear_table.py'

def summarize():
 expected={c['id']:c for c in execution.catalog.catalog()};pr=execution.read(execution.record(execution.META+'/prerequisite.json'));records=[];phases=[]
 for r in pr['reused_case_records']:records.append((r['source_result'],r['external_cached_result'],True))
 cache=json.loads((ROOT/execution.META/'terminal-case-cache.json').read_text());cached={x['path']:x for x in cache['records']}
 for i in range(3):
  protocol=execution.META+f'/gpu{i}/protocol-v02.json';p,_=execution.validate(protocol,execution.worker.sha(ROOT/protocol));f=ROOT/p['output']/'execution.json';d=json.loads(f.read_text())
  if d.get('terminal') is not True:raise ValueError('Wait until all phases terminal')
  if Path('/proc/'+str(d['pid'])).exists():raise ValueError('Controller still live; reap before completion seal')
  phases.append(dict(phase=p['phase_id'],execution=execution.worker.record(f),completed=d['completed'],passed=d['passed'],queued_case_ids=d['queued_case_ids'],seconds=d['seconds']))
  for r in d['records']:
   if not r.get('terminal') or Path('/proc/'+str(r['pid'])).exists():raise ValueError('Worker not terminal')
   if not r.get('result'):continue
   c=cached.get(r['result']['path'])
   if c is None or c['sha256']!=r['result']['sha256']:raise ValueError('External terminal case cache not ready')
   records.append((r['result'],dict(path=c['cache'],sha256=c['sha256']),False))
 rows=[]
 for source,cache,reused in records:
  result=execution.read(source);cached=execution.read(cache)
  if source['sha256']!=cache['sha256'] or result!=cached:raise ValueError('Case/cache mismatch')
  case=result['case'];e=expected.get(case['id'])
  if e is None or any(case[k]!=e[k] for k in ['young_Pa','support','Fz_N']):raise ValueError('Unexpected physical case')
  meshes=[]
  for m in result.get('meshes',[]):
   if tuple(m['cells']) not in execution.worker.MESHES:continue
   if m.get('capture'):execution.catalog.checked_record(m['capture'])
   diag=m.get('diagnostics',{})
   meshes.append(dict(cells=m['cells'],capture=m.get('capture'),completed=m.get('completed'),physics_passed=m.get('physics_passed'),original_R24_N=m.get('original_free_R24_N'),original_R80_N=m.get('original_free_R80_N'),minimum_whole_J=m.get('whole_P2_J',{}).get('minimum_lower_bound'),bulk_nonrigid_coordinate_RMS_m=diag.get('bulk_nonrigid_coordinate_RMS_m'),peak_principal_strain=diag.get('maximum_absolute_principal_stretch_strain'),volume99_principal_strain=diag.get('volume_weighted99th_percentile_max_abs_principal_strain'),seconds=m.get('seconds')))
  if result.get('completed') and (len(meshes)!=2 or {tuple(m['cells']) for m in meshes}!=set(execution.worker.MESHES)):raise ValueError('Completed case requires exact32/40 field pair')
  rows.append(dict(case=e,source_result=source,external_cache=cache,reused=reused,completed=result.get('completed',False),physics_passed=result.get('physics_passed',False),passed=result.get('passed',False),conditional_spatial_indicators_m=result.get('conditional_spatial_indicators_m',[]),meshes=meshes))
 if len(rows)!=len({x['case']['id'] for x in rows}):raise ValueError('Duplicate cases')
 missing=sorted(set(expected)-{x['case']['id'] for x in rows});complete=not missing and all(x['completed'] for x in rows) and all(x['completed'] for x in phases)
 report=dict(schema='C38-static-nonlinear-forty-state-terminal-v01',completed=complete,all40_declared_q63_static_gates_passed=complete and all(x['passed'] and x['physics_passed'] for x in rows),counts=dict(planned_states=40,completed_states=sum(x['completed'] for x in rows),reused_states=sum(x['reused'] for x in rows),new_terminal_result_records=sum(not x['reused'] for x in rows)),source=execution.record(SOURCE),prerequisite=execution.record(execution.META+'/prerequisite.json'),phases=phases,missing_result_case_ids=missing,rows=rows,observations_roles_noise_or_training_drawn=False,interpolation_accuracy_evaluated=False,continuous_material_or_whole_field_qualification=False,dynamic_contact_image_inference_qualification=False,scope='Each declared state uses original mechanics and conditional actual-h63-point displacement indicator; assumedp>=1 is not a rigorous bound. Local clamp stress is not qualified.',raw_arrays_loaded=False)
 return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(execution.fresh_write(a.output,summarize())))
