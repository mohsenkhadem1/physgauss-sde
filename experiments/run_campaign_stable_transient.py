"""Prospective BE/BDF2 fixed-mesh physical-time refinement, no NewtonGPUcalls."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_stable_transient import StableBlockTransient
from experiments.run_static_supported_block import probes
from experiments.diagnose_nonlinear_physical_probes import interpolate
from experiments.run_static_p2_axial_continuation import sha,dump,record
RATES=(240,480,960,1920);METHODS=('BE','BDF2')
def prepare(path):
 sources=['experiments/campaign_stable_transient.py','experiments/run_campaign_stable_transient.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/run_static_supported_block.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py','experiments/run_static_p2_axial_continuation.py']
 d={'schema':'StableNH-physical-transient-protocol-v0.1','status':'FROZEN_BEFORE_TRANSIENT_TRAJECTORIES','CPU_threads':2,'hard_wall_cap_s':300,'methods':list(METHODS),'rates_hz':list(RATES),'duration_s':.1,'output_query_hz':240,'output_query_frames':25,'model':{'cells':[4,2,2],'tets':96,'size_m':[.24,.08,.08],'E_Pa':5000.,'nu':.35,'rho_kg_m3':625.,'body_mass':'ExactP1tetvolume/nodelumping,retainedatfixednodes','boundary':'Fullleftfaceallaxesexactfixed','gravity':[0,0,0],'physical_damping':0.,'force':'1Nstepatt0helduniformreferencerighttractionall720steps','initial':'REST+zeroV,freshforall8trajectories','law':'Newton1.7VBDstableNHkappa=lambdaL+muL'},'BE':{'h_eff':'h','target':'qn+h*vn','velocity':'(qnew-qn)/h'},'BDF2':{'startup':'oneBEphysicalstep','h_eff':'2h/3','target':'(4qn-qprev)/3+(2h/9)*(4vn-vprev),evaluatedstableincrements','velocity':'(3qnew-4qn+qprev)/(2h),evaluatedstableincrements','physical_residual':'M*(3vnew-4vn+vprev)/(2h)+gradE(qnew)-F'},'solver':{'objective':'E(q)+0.5sumM/h_eff²*(q-y)²-F·q','strict_objective_free_R_N':1e-10,'original_physical_free_R_gate_N':1e-9,'maximum_Newton_iterations':20,'line_search':'PositiveJ+originalobjectiveArmijo,unmodifiedtangent/no regularization','initialiterate':'acceptedqn,notdistantforcepredictor','no_retry_or_physics_retune':True},'gates':{'minimum_J':.1,'fixed_constraint_error_m':0.,'temporal_finest_adjacent_query_max_Euclidean_m':20e-6,'adjacent_differences_nonincreasing':True,'all3adjacent_errors_retained':True,'coarse_failures_not_overwritten':True},'probe_groups':{k:v.tolist() for k,v in probes().items()},'source_sha256':{f:sha(f) for f in sources},'limits':['PhysicaltimeBE/BDF2compareonSAME96tetstablelaw','TemporalqualificationONLY;P1spatialerrorunresolved','NoGPU/nativeaccuracy/contact/calibrationinferenceclaim','ZeroDirichletvelocityexact;BDF2physicalvelocityNOTnativehelpervelocity']}
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 dump(p,d)
def run(protocol,digest,output):
 if sha(protocol)!=digest:raise ValueError('Changed protocol')
 p=json.loads(Path(protocol).read_text())
 for f,h in p['source_sha256'].items():
  if sha(f)!=h:raise ValueError('Changed source '+f)
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();points=np.concatenate([np.asarray(v) for v in p['probe_groups'].values()]);d={'schema':'StableNH-physical-transient-results-v0.1','completed':False,'protocol':record(protocol),'cases':[],'comparisons':[],'temporal_qualification':{},'spatial_or_native_or_inference_qualified':False,'planned':[{'method':m,'hz':r,'status':'QUEUED'} for m in METHODS for r in RATES]};series={}
 def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
 signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Frozen300sCPU2transientbudget')));signal.alarm(300)
 try:
  for method in METHODS:
   for hz in RATES:
    begin=time.monotonic();row={'method':method,'hz':hz,'completed':False};slot=next(r for r in d['planned'] if r['method']==method and r['hz']==hz);slot['status']='RUNNING';save();model=StableBlockTransient();state=model.initial();prev=None;qs=[state.q.copy()];vs=[state.v.copy()];ledger=[];steps=int(hz/10)
    try:
     for n in range(1,steps+1):
      new,check=model.step(state,1/hz,method,prev);ledger.append({'step':n,'interval_s':[(n-1)/hz,n/hz],**check})
      if check['physical_original_free_R_N']>1e-9 or check['minimum_J']<=.1 or check['constraint_defect_m']!=0 or not np.isfinite(new.v).all():raise RuntimeError('PhysicaloriginalR/J/clamp/finite gatefailed')
      prev,state=state,new;qs.append(state.q.copy());vs.append(state.v.copy())
     times=np.arange(steps+1)/hz;qs=np.asarray(qs);vs=np.asarray(vs);query=interpolate(model.model.rest,model.model.tets,qs[::hz//240],points)[0];file=out/f'{method}-hz{hz}.npz'
     np.savez_compressed(file,times_s=times,positions_m=qs,velocities_m_s=vs,query_times_s=np.arange(25)/240,query_positions_m=query,material_probes_m=points,rest_m=model.model.rest,tets=model.model.tets,faces=model.model.faces,physical_masses_kg=model.mass,constraint_mask=model.mask,applied_forces_N=model.force)
     row.update(completed=True,steps=steps,frames=len(times),capture=record(file),maximum_objective_free_R_N=max(x['objective_free_R_N'] for x in ledger),maximum_physical_original_free_R_N=max(x['physical_original_free_R_N'] for x in ledger),minimum_J=min(x['minimum_J'] for x in ledger),maximum_constraint_defect_m=max(x['constraint_defect_m'] for x in ledger),maximum_Newton_iterations=max(x['iterations'] for x in ledger),physical_force_impulse_Ns=(model.force.sum(0)*.1).tolist());series[(method,hz)]=(query,qs[::hz//240])
     if hz//2 in RATES and(method,hz//2) in series:
      oldq,oldnode=series[(method,hz//2)];delta=np.linalg.norm(query-oldq,axis=2);d['comparisons'].append({'method':method,'rates_hz':[hz//2,hz],'full25frame_query_max_Euclidean_m':float(delta.max()),'full25frame_query_coordinate_RMS_m':float(np.sqrt(np.mean((query-oldq)**2))),'full25frame_node_max_Euclidean_m':float(np.linalg.norm(qs[::hz//240]-oldnode,axis=2).max()),'passed20um':bool(delta.max()<=20e-6)})
     slot['status']='COMPLETED'
    except TimeoutError as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='BUDGET_INTERRUPTED';raise
    except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED'
    finally:
     row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/f'{method}-hz{hz}.json',row);dump(out/f'{method}-hz{hz}-steps.json',ledger);save()
  for method in METHODS:
   pairs=[r for r in d['comparisons'] if r['method']==method];errors=[r['full25frame_query_max_Euclidean_m'] for r in pairs];allcases=[r for r in d['cases'] if r['method']==method];monotone=len(errors)==3 and all(b<=a for a,b in zip(errors,errors[1:]));passed=len(allcases)==4 and all(r['completed'] for r in allcases) and monotone and pairs[-1]['passed20um']
   d['temporal_qualification'][method]={'all4_rates_completed':all(r['completed'] for r in allcases),'all_adjacent_query_errors_m':errors,'nonincreasing':monotone,'finest_adjacent20um_passed':bool(len(errors)==3 and pairs[-1]['passed20um']),'fixed_mesh_temporal_passed':passed}
  d['completed']=True
 except Exception as e:d['execution_failure']=repr(e)
 finally:signal.alarm(0);save()
 return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
 if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
 else:
  d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','temporal_qualification','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
