"""Terminal-only compact evidence join; no physical array loads or solves."""
from pathlib import Path
import hashlib,json,os,subprocess
import numpy as np
from experiments import run_campaign_c38_graded_family_warp as m
ROOT=m.ROOT;OP=ROOT/'runs/operator-dev/c38-graded-family-warp-v01';OUT=ROOT/'runs/gaussian-campaign-v01/c38-graded-family-warp-v01'
PROTO=OP/'protocol-v01.json';DIGEST='9ed421d31357de2d384564354e72629ba32a0c46e5f5ca75c42d439ac42a960c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def absent(pid):
 try:os.kill(pid,0)
 except ProcessLookupError:return True
 return False
p=m.validate(PROTO,DIGEST);e=load(OUT/'execution.json')
assert e['completed'] is True and e['passed'] is True and len(e['records'])==12 and not e['queued_case_ids'] and e['first_case_gate_passed']
assert all(r['terminal'] and r['exit_code']==0 and r['completed'] for r in e['records'])
pids=[e['pid']]+[r['pid'] for r in e['records']];assert all(absent(pid) for pid in pids)
cache=load(OP/'completed-case-cache-receipt.json');cache_by_sha={r['sha256']:r for r in cache['records']};assert len(cache_by_sha)==12
expected={r['id']:r for r in p['all_original_cases']};sources=[]
for r in e['records']:
 record=r['result'];assert sha(ROOT/record['path'])==record['sha256'];c=cache_by_sha[record['sha256']];assert sha(ROOT/c['cache'])==record['sha256'];sources.append((record,False))
for record in p['reused_completed_case_results']:sources.append((record,True))
assert len(sources)==14
rows=[];captures=[];new_linear_count=0;original_linear_count=0;selected_protocols={}
for source,reused in sources:
 assert sha(ROOT/source['path'])==source['sha256'];s=load(ROOT/source['path']);case=s['case'];assert m.physical_key(case)==m.physical_key(expected[case['id']]);assert s['completed'] and s['passed'] and s['physics_passed'] and absent(s['pid'])
 if reused:
  assert sha(ROOT/s['protocol']['path'])==s['protocol']['sha256'];sp=load(ROOT/s['protocol']['path']);selected_protocols[s['protocol']['path']]=s['protocol']['sha256']
  for name,h in sp['source_sha256'].items():assert sha(ROOT/name)==h
 meshes=sorted([v for v in s['meshes'] if v['cells'][0] in (32,40)],key=lambda v:v['cells'][0]);assert len(meshes)==2 and [v['cells'][0] for v in meshes]==[32,40]
 compact=[]
 for v in meshes:
  assert v['completed'] and v['physics_passed'];d=v['diagnostics'];assert v['original_free_R24_N']<=1e-10 and v['original_free_R80_N']<=1e-10 and v['gradient24_80_difference_N']<=1e-10 and v['energy24_80_difference_J']<=1e-11
  assert d['force_balance_defect_N']<=1e-8 and d['actual_moment_balance_defect_Nm']<=1e-8 and d['maximum_constraint_error_m']==0 and d['minimum_J']>.1 and v['whole_P2_J']['all_elements_positive']
  assert sha(ROOT/v['capture']['path'])==v['capture']['sha256'];captures.append(dict(case_id=case['id'],cells=v['cells'],capture=v['capture'],reused_exact_field=reused))
  if reused:
   assert v['linear_records']
   for line in v['linear_records']:assert line['original_true_residual_N']<=max(2*line['requested_tolerance_N'],1e-11)
   original_linear_count+=len(v['linear_records'])
  else:
   checks=v['independent_linear_checks'];assert checks and v['all_independent_original_Torch_linear_checks_passed'] and all(c['passed'] and c['independent_original_Torch_HVP_true_R_N']<=c['original_CG_acceptance_bound_N']+c['float64_evaluation_margin_N'] for c in checks);new_linear_count+=len(checks)
  compact.append(dict(cells=v['cells'],capture=v['capture'],hmax_m=v['hmax_m'],original_R24_N=v['original_free_R24_N'],original_R80_N=v['original_free_R80_N'],gradient24_80_difference_N=v['gradient24_80_difference_N'],energy24_80_difference_J=v['energy24_80_difference_J'],force_balance_defect_N=d['force_balance_defect_N'],actual_moment_balance_defect_Nm=d['actual_moment_balance_defect_Nm'],sampled_minimum_J=d['minimum_J'],whole_element_J_lower_bound=v['whole_P2_J']['minimum_lower_bound'],sampled_peak_principal_strain=d['maximum_absolute_principal_stretch_strain'],volume99_principal_strain=d['volume_weighted99th_percentile_max_abs_principal_strain'],volume_principal_strain_RMS=d['volume_weighted_principal_strain_RMS'],bulk_nonrigid_coordinate_RMS_m=d['bulk_nonrigid_coordinate_RMS_m'],seconds=v['seconds']))
 assert len(s['conditional_spatial_indicators_m'])>=1
 indicator=s['conditional_spatial_indicators_m'][-1];assert indicator<=20e-6
 rows.append(dict(case=case,case_result=source,reused_exact_selected_case=reused,meshes=compact,conditional_actual_h_q63_indicator_m=indicator,conditional_displacement_passed=True,finest=compact[-1],seconds=s['seconds']))
assert len({r['case']['id'] for r in rows})==14 and len(captures)==28 and sum(c['reused_exact_field'] for c in captures)==4
rows.sort(key=lambda r:list(expected).index(r['case']['id']))
original=load(ROOT/m.METADATA_INPUTS['original14_summary']);selected=load(ROOT/m.METADATA_INPUTS['selected_summary'])
summary=dict(schema='C38-completed-finite14-graded-q63-v01',completed=True,all14_finite_case_q63_displacement_indicators_passed=True,counts=dict(cases=14,new_cases=12,reused_cases=2,new_fine_fields=24,reused_fine_fields=4,total_fine_fields=28),scope='Finite14 declared engineering configurations; conditional actual-h displacement indicator at the SAME63 fixed material points ONLY',continuous_E_uniform_domain_certificate=False,whole_field_or_surface_pixel_spatial_qualification=False,peak_stress_or_local_strain_qualification=False,static_dataset_or_image_admission=False,observations_roles_or_training_generated=False,protocol=rec(PROTO),execution=rec(OUT/'execution.json'),seconds_new12=e['seconds'],rows=rows,independent_original_Torch_Warp_direction_checks=new_linear_count,reused_original_Torch_CG_residual_checks=original_linear_count,preserved_uniform_summary=rec(ROOT/m.METADATA_INPUTS['original14_summary']),preserved_uniform_spatial_failures=7,preserved_selected24_to32_failures=selected['retained_failed_comparisons'],original_full_prior_case_JSON_recovered=False,metadata_loss_statement=selected['prior_full_report_loss'],metadata_lineage=selected['metadata_lineage'],peak_warning='Local edge/clamp peaks grow under refinement; displacement PASS does not qualify converged stress, moderate local strain or material realism.',raw_local=True)
for name,key in [('sampled_peak_strain_range','sampled_peak_principal_strain'),('volume99_strain_range','volume99_principal_strain'),('volume_RMS_strain_range','volume_principal_strain_RMS'),('bulk_nonrigid_RMS_m_range','bulk_nonrigid_coordinate_RMS_m'),('sampled_minimum_J_range','sampled_minimum_J'),('whole_element_J_lower_bound_range','whole_element_J_lower_bound')]:
 values=[r['finest'][key] for r in rows];summary[name]=[min(values),max(values)]
summary['q63_indicator_m_range']=[min(r['conditional_actual_h_q63_indicator_m'] for r in rows),max(r['conditional_actual_h_q63_indicator_m'] for r in rows)]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
gpu2=[line for line in apps.splitlines() if p['resources']['physical_GPU_UUID'] in line];assert not gpu2
proof=dict(schema='C38-remaining12-terminal-proof-v01',completed=True,protocol=rec(PROTO),execution=rec(OUT/'execution.json'),controller_exit_code=0,controller_reaped_via_unified_exec_session=75768,owned_PID_absence={str(pid):absent(pid) for pid in pids},all_owned_PIDs_absent=True,GPU2_compute_idle=True,source_sha256=p['source_sha256'],all37_sources_unchanged=True,reused_source_protocol_sha256=selected_protocols,all28_capture_hashes_match=True,captures=captures,all12_complete_case_bytes_cached=True,completed_case_cache=rec(OP/'completed-case-cache-receipt.json'),all24_new_mesh_physics_and_independent_CG_checks_passed=True,all4_reused_original_Torch_CG_mesh_checks_passed=True,original_lost_report_not_recovered=True,no_new_physics_or_array_queries=True,raw_local=True)
save(OP/'completed-summary.json',summary);save(OP/'terminal-proof.json',proof)
print(json.dumps({'summary':rec(OP/'completed-summary.json'),'terminal_proof':rec(OP/'terminal-proof.json'),'q63_indicator_um_range':[x*1e6 for x in summary['q63_indicator_m_range']],'peak_strain_range':summary['sampled_peak_strain_range'],'seconds':e['seconds'],'GPU2_compute_idle':True}))
