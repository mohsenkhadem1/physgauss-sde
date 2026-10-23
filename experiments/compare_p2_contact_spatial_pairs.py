"""Prospective read-only common-material-query comparison, no physics solves."""
import argparse,json,os,signal
from pathlib import Path
import numpy as np
from experiments import run_p2_contact_spatial_pairs as pair
ROOT=pair.ROOT;META=pair.META
SOURCE='experiments/compare_p2_contact_spatial_pairs.py';TEST='tests/test_compare_p2_contact_spatial_pairs.py'
TIGHT96=ROOT/'runs/gaussian-campaign-v01/p2-corrected-contact-order128-trajectory-v01/outer128_accuracy_1e10/trajectory.npz'

def contraction(first,second):return bool(second<first or first==second==0.)
def combine(a,b):
 q=contraction(a['maximum_common81_q_difference_m'],b['maximum_common81_q_difference_m']);v=contraction(a['maximum_common81_v_difference_m_s'],b['maximum_common81_v_difference_m_s'])
 return dict(q_errors_contract=q,v_errors_contract=v,finest_conditional_q_gate_met=bool(q and b['q_passed']),finest_conditional_v_gate_met=bool(v and b['v_passed']),fixed_root128_operator_only=True,continuum_or_contact_dataset_admitted=False)

def prepare():
 phases={}
 for name in pair.PHASES:
  path=META/name/'protocol-v02.json';phases[name]=dict(path=str(path.relative_to(ROOT)),sha256=pair.sha(path));pair.validate(phases[name]['path'],phases[name]['sha256'])
 p=dict(schema='contact-root128-common81-spatial-comparison-v01',status='FROZEN_BEFORE_FINE_COMPARISON_NO_ARRAY_READS',phase_protocols=phases,source_sha256={SOURCE:pair.sha(ROOT/SOURCE),TEST:pair.sha(ROOT/TEST)},imported96tight=dict(path=str(TIGHT96.relative_to(ROOT)),sha256=pair.sha(TIGHT96)),fine_case='accuracy_1e10',common_queries='Exact81materialpoints/times storedbyfrozenworkers; P2queryphysicalvelocity aswellaspositions. NEVERallnodecrossmeshcomparison.',old96_velocity_interpolation='Oldcapture lacks queryvelocity: originalauthored96P2build/rest/elementsidentity, query(savedphysicalv,exactmaterialpoints). No nearestnodes/resimulation.',gates=dict(raw_q_m=20e-6,raw_v_m_s=1e-4,conditional_q_m=20e-6,conditional_v_m_s=1e-4,adjacent_indicator_factors=[2.,3.],q_and_v_raw_errors_must_contract=True,finest_pass_requires_finest_conditional_gate_and_contraction=True),all_adjacent_results_retained=True,CPU_affinity_cores=[0,1],hard_cap_s=60,limits=['Separate temporal andspatial gates; no combined20umerrorbound','Finestconditionalindicator assumesp>=1, NOT rigorouscontinuumbound','Allroot128meshresults arefixed-surface-operator comparisons; no continuum/hardfloor/cohortadmission','Allfinepairs mustbeTERMINALbeforecomparison; failuresretained, no redraw/retry/selection'])
 path=META/'spatial-comparison-protocol.json'
 if path.exists():raise FileExistsError('Preserve prospective protocol')
 pair.write(path,p);return pair.sha(path)

def run(digest):
 path=META/'spatial-comparison-protocol.json'
 if pair.sha(path)!=digest:raise ValueError('Changedprotocol')
 p=json.loads(path.read_text())
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!=set(p['CPU_affinity_cores']):raise ValueError('CPU2ONLY')
 if any(pair.sha(ROOT/x)!=h for x,h in p['source_sha256'].items()):raise ValueError('Changedcomparisonsource')
 phase_results=[];evidence=[];data=[]
 for name,record in p['phase_protocols'].items():
  config=pair.validate(record['path'],record['sha256']);summary=ROOT/config['output']/'summary.json';done=json.loads(summary.read_text())
  if not done.get('terminal') or not done.get('completed'):raise ValueError('Bothfinepairs mustcomplete BEFOREarrays')
  phase_results.append(done);evidence.append(dict(path=str(summary.relative_to(ROOT)),sha256=pair.sha(summary)))
 def cap(*args):raise TimeoutError('FrozenCPU60s comparison cap')
 signal.signal(signal.SIGALRM,cap);signal.alarm(60)
 try:
  if pair.sha(TIGHT96)!=p['imported96tight']['sha256']:raise ValueError('Changedimport96')
  from experiments.run_p2_contact_validation import build
  m=build([4,2,2],300000.,9.81)
  with np.load(TIGHT96,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['rest_m'],m.rest);np.testing.assert_array_equal(z['elements'],m.elements)
   a={k:z[k].copy() for k in ['times_s','material_probes_m','query_positions_m']};a['query_velocities_m_s']=np.array([m.query(v,a['material_probes_m']) for v in z['velocities_m_s']]);data.append(a)
  for done in phase_results:
   row=next(r for r in done['cases'] if r['config']['name']==p['fine_case']);r=row['capture']
   if pair.sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changedfinecrafture')
   with np.load(ROOT/r['path'],allow_pickle=False) as z:data.append({k:z[k].copy() for k in ['times_s','material_probes_m','query_positions_m','query_velocities_m_s']})
   evidence.append(r)
  comparisons=[pair.spatial_pair(data[0],data[1],2.),pair.spatial_pair(data[1],data[2],3.)]
  summary=dict(completed=True,protocol=dict(path=str(path.relative_to(ROOT)),sha256=digest),source_sha256=p['source_sha256'],evidence=evidence,imported96tight=p['imported96tight'],adjacent_meshes=[[96,324],[324,768]],comparisons=comparisons,**combine(*comparisons),separate_same_mesh_time_controls=[dict(mesh=name,time_reference_qualified=done['time_reference_qualified'],accuracy_pair=done['accuracy_pair']) for name,done in zip(p['phase_protocols'],phase_results)],fixed_material_query_count=81,common_clock_count=len(data[0]['times_s']),all_errors_unfiltered=True,no_new_physics=True)
  pair.write(META/'spatial-comparison.json',summary);return summary
 finally:signal.alarm(0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();print(json.dumps(prepare() if a.action=='prepare' else run(a.sha)))
