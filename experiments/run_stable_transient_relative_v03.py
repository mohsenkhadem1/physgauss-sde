"""Prospective numeric-only3840parity/7680 refinement, oldfailurepreserved."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_stable_transient_relative_v03 import RelativeStableTransientV03 as RelativeStableTransient
from experiments.diagnose_nonlinear_physical_probes import interpolate
from experiments.run_static_p2_axial_continuation import sha,dump,record,checked_record
OLD=Path('runs/gaussian-campaign-v01/stable-transient-refinement-v01')
def prepare(path):
 d=checked_record(record(OLD/'summary.json'));oldp=checked_record(d['protocol']);r=next(r for r in d['cases'] if r['hz']==3840)
 if not r['completed'] or not any(x['hz']==7680 and not x['completed'] for x in d['cases']):raise ValueError('Preservedprecisionfailure/3840reference missing')
 sources=dict(oldp['source_sha256']);sources.update({f:sha(f) for f in ['experiments/campaign_stable_transient_relative.py','experiments/campaign_stable_transient_relative_v03.py','experiments/run_stable_transient_relative_v03.py']})
 p={'schema':'StableBDF2-numeric-relative-protocol-v0.3','status':'FROZEN_BEFORE_NEW_TRAJECTORIES','old_summary':record(OLD/'summary.json'),'old_protocol':d['protocol'],'old3840':r['capture'],'old1920':oldp['old1920'],'preserved_v02_failure':record('runs/gaussian-campaign-v01/stable-transient-relative-v02/summary.json'),'source_sha256':sources,'rates_hz':[3840,7680],'CPU_threads':2,'hard_wall_cap_s':120,'numeric_change':'CompatibleF=I+u*restgradients;EXACTfrozenstable_material;samemathematicalE/P/A. Solvezwithsameobjective. Armijoroundoff128eps*max(EyoungV,absE,absV),strictRdecreaseonroundoffbranch. Finaltolerancesunchanged','unchanged_gates':{'objective_free_R_N':1e-10,'physical_original_free_R_N':1e-9,'minimum_J':.1,'constraint_m':0.,'finest_adjacent_query_error_m':20e-6,'all5_adjacent_nonincreasing':True},'new_parity_guard':{'3840_same_frozen_completed_query_max_m':1e-8,'world_displacement_roundtrip_m':1e-14},'world_chord_R':'Reportindependentlyfromsavedglobalq,NOTaninternalstationaritygate;globalqroundoffisexplicitandboundedbyparity/roundtripguards','methods':'BDF2physicalfirstordersystem;firststepBE;forcelawclocktolerancesunchanged','duration_s':.1,'no_retry_or_extra_rates':True,'limits':['Old7680failurepreserved','Fixed96tettemporalonly,notspatial/native/inferencequalification']}
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 dump(path,p)
def run(protocol,digest,output):
 if sha(protocol)!=digest:raise ValueError('Changed protocol')
 p=json.loads(Path(protocol).read_text());old=checked_record(p['old_summary']);checked_record(p['old_protocol']);checked_record(p['preserved_v02_failure'])
 for f,h in p['source_sha256'].items():
  if sha(f)!=h:raise ValueError('Changed source '+f)
 for name in ['old3840','old1920']:
  if sha(p[name]['path'])!=p[name]['sha256']:raise ValueError('Changedoldcapture')
 with np.load(p['old1920']['path'],allow_pickle=False) as a:points=a['material_probes_m'].copy();prior=a['query_positions_m'].copy()
 with np.load(p['old3840']['path'],allow_pickle=False) as a:old3840=a['query_positions_m'].copy()
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();d={'schema':'StableBDF2-relative-results-v0.3','completed':False,'protocol':record(protocol),'cases':[],'comparisons':[],'fixed_mesh_temporal_qualified':False,'old7680failure_preserved':True,'spatial_or_GPU_or_inference_qualified':False};history=checked_record(checked_record(old['protocol'])['old_summary'])['temporal_qualification']['BDF2']['all_adjacent_query_errors_m'].copy();priorhz=1920
 def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
 signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Frozen120srelativebudget')));signal.alarm(120)
 try:
  for hz in [3840,7680]:
   begin=time.monotonic();row={'method':'BDF2','hz':hz,'completed':False};m=RelativeStableTransient();s=m.initial();prev=None;qs=[m.model.rest+s.u];us=[s.u.copy()];vs=[s.v.copy()];zs=[s.delta.copy()];ledger=[];steps=hz//10
   try:
    for n in range(1,steps+1):
     new,check=m.step(s,1/hz,'BDF2',prev);ledger.append({'step':n,'interval_s':[(n-1)/hz,n/hz],**check})
     if check['physical_original_free_R_N']>1e-9 or check['minimum_J']<=.1 or check['constraint_defect_m']!=0 or check['maximum_world_displacement_roundtrip_m']>1e-14 or not np.isfinite(new.v).all():raise RuntimeError('Unchangedphysical/newroundtripgatefailed')
     prev,s=s,new;qs.append(m.model.rest+s.u);us.append(s.u.copy());vs.append(s.v.copy());zs.append(s.delta.copy())
    qs=np.asarray(qs);vs=np.asarray(vs);query=interpolate(m.model.rest,m.model.tets,qs[::hz//240],points)[0];worldv=np.zeros_like(qs);worldv[1]=(qs[1]-qs[0])*hz;worldv[2:]=(3*(qs[2:]-qs[1:-1])-(qs[1:-1]-qs[:-2]))*hz/2;world_R=[]
    for n in range(1,len(qs)):
     acc=(worldv[n]-worldv[n-1])*hz if n==1 else(3*worldv[n]-4*worldv[n-1]+worldv[n-2])*hz/2
     R=m.mass[:,None]*acc+m.model.elastic(qs[n])[1]-m.force;world_R.append(float(np.linalg.norm(R[~m.mask])))
    parity=float(np.linalg.norm(query-old3840,axis=2).max()) if hz==3840 else None
    if parity is not None and parity>1e-8:raise RuntimeError('Prospective3840paritygatefailed')
    file=out/f'BDF2-hz{hz}.npz';np.savez_compressed(file,times_s=np.arange(steps+1)/hz,positions_m=qs,displacements_m=np.asarray(us),step_increments_m=np.asarray(zs),velocities_m_s=vs,world_chord_velocities_m_s=worldv,query_times_s=np.arange(25)/240,query_positions_m=query,material_probes_m=points,rest_m=m.model.rest,tets=m.model.tets,faces=m.model.faces,physical_masses_kg=m.mass,constraint_mask=m.mask,applied_forces_N=m.force)
    row.update(Armijo_energy_scale_J=float(m.model.young*m.model.volumes.sum()),maximum_Armijo_roundoff_window_J=max(x['maximum_Armijo_roundoff_window_J'] for x in ledger),completed=True,steps=steps,capture=record(file),maximum_objective_free_R_N=max(x['objective_free_R_N'] for x in ledger),maximum_physical_original_free_R_N=max(x['physical_original_free_R_N'] for x in ledger),minimum_J=min(x['minimum_J'] for x in ledger),maximum_constraint_defect_m=max(x['constraint_defect_m'] for x in ledger),maximum_world_displacement_roundtrip_m=max(x['maximum_world_displacement_roundtrip_m'] for x in ledger),maximum_world_chord_roundoff_m=max(x['maximum_world_chord_roundoff_m'] for x in ledger),maximum_u_chord_roundoff_m=max(x['maximum_u_chord_roundoff_m'] for x in ledger),maximum_saved_world_chord_R_N=max(world_R),same3840_reference_parity_m=parity)
    error=float(np.linalg.norm(query-prior,axis=2).max());d['comparisons'].append({'rates_hz':[priorhz,hz],'full25frame_query_max_Euclidean_m':error,'passed20um':error<=20e-6});history.append(error);prior,priorhz=query,hz
   except TimeoutError as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};raise
   except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])}
   finally:row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/f'BDF2-hz{hz}.json',row);dump(out/f'BDF2-hz{hz}-steps.json',ledger);save()
  d['completed']=True;d['all5_adjacent_query_errors_m']=history;d['nonincreasing']=len(history)==5 and all(b<=a for a,b in zip(history,history[1:]));d['fixed_mesh_temporal_qualified']=all(r['completed'] for r in d['cases']) and d['nonincreasing'] and history[-1]<=20e-6
 except Exception as e:d['execution_failure']=repr(e)
 finally:signal.alarm(0);save()
 return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
 if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
 else:
  d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','fixed_mesh_temporal_qualified','comparisons','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
