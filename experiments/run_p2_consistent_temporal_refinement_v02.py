"""Prospective bounded continuation; immutable prior failures and core retained."""
import argparse,json,os,time,signal,multiprocessing
from pathlib import Path
import numpy as np
from experiments.campaign_p2_consistent_transient import ConsistentP2Transient,force_coefficient
from experiments.run_p2_consistent_transient import query_operator
from experiments.run_static_p2_axial_continuation import sha,dump,record,checked_record,watchdog
BASE=Path('runs/gaussian-campaign-v01/p2-consistent-transient-v01')


def eligible_larger_case(errors,elapsed,forecast,cap=300.):
    return bool(len(errors)>=2 and errors[-1]<=20e-6 and errors[-1]<=errors[-2] and elapsed+forecast+5<cap)


def prepare(path):
    old=json.loads((BASE/'summary.json').read_text());rows=old['cases'];ramp=next(r for r in rows if r['id']=='tets96-smooth_ramp-hz480' and r['completed'])
    refs=[next(r for r in rows if r['id']==f'tets6-smooth_ramp-hz{hz}' and r['completed'])['capture'] for hz in [240,480,960]]
    sources=['experiments/run_p2_consistent_temporal_refinement_v02.py','experiments/campaign_p2_consistent_transient.py','experiments/run_p2_consistent_transient.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/campaign_stable_transient.py','experiments/run_static_supported_block.py','experiments/run_static_p2_axial_continuation.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    p={'schema':'C38-P2-smooth-temporal-refinement-protocol-v0.1','status':'FROZEN_BEFORE_REFINED_TIME_FIELDS','orchestration_revision':'v02 hash-only binary reference validation','preserved_preflight_failure_protocol':record(Path('runs/operator-dev/p2-consistent-temporal-refinement-v01/protocol.json')),'source_sha256':{f:sha(f) for f in sources},'prior_protocol':old['protocol'],'prior_summary':record(BASE/'summary.json'),'six_tet_references':dict(zip(map(str,[240,480,960]),refs)),'larger_reference480':ramp['capture'],'larger_cost_basis_s':ramp['elapsed_s'],'larger_runtime_forecast_with10percent_safety_s':1.1*4*ramp['elapsed_s'],'CPU_threads':2,'hard_wall_cap_s':300,'actual_RSS_guard_bytes':8*1024**3,'required_jobs':[{'id':f'tets6-smooth_ramp-hz{hz}','cells':[1,1,1],'hz':hz,'tetrahedra':6} for hz in [1920,3840]],'conditional_job':{'id':'tets96-smooth_ramp-hz1920','cells':[4,2,2],'hz':1920,'tetrahedra':96},'conditional_rule':'6tet finest1920-to3840<=20um AND contracts, AND forecast233.19s+5s fits remaining300s; otherwise preserve SKIPPED','physics_and_scheme':'EXACT prior protocol; unchanged stableNH/full consistent T10 M/exactleftclamp/E5000/nu.35/rho625/gravity0/damp0/1Nrightface;BE-startBDF2','program':'smooth_ramp coefficient sin²((pi/2)*min(t/0.05,1)), endpointtime(n+1)/Hz, duration0.1s','gates':{'objective_R_N':1e-10,'physical_R_N':1e-9,'minimum_sampled_J':.1,'clamp_position_and_velocity':0.,'full25times122probes_finest_adjacent_max_m':20e-6,'positive_contraction':True},'limits':['All original coarse failures and wall timeout immutable','No retry unchanged missing960 larger run','96tet480-to1920 is a coarse/fine comparison with no successive-level contraction proof','No spatial/native/training/inference qualification']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,p)


def verify_record(r):
    if sha(r['path'])!=r['sha256']:raise ValueError('Changed pinned input '+r['path'])


def validate(path,digest):
    if sha(path)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(path).read_text())
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed frozen source '+f)
    for k in ['prior_protocol','prior_summary','larger_reference480','preserved_preflight_failure_protocol']:verify_record(p[k])
    for r in p['six_tet_references'].values():verify_record(r)
    return p


def run(path,digest,output):
    p=validate(path,digest)
    for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        if os.environ.get(var)!='2':raise ValueError('Explicit two CPU threads required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();s={'schema':'C38-P2-smooth-temporal-refinement-results-v0.1','completed':False,'protocol':record(path),'cases':[],'prior_failures_preserved':True,'six_tet_temporal_passed':False,'larger_temporal_qualified':False,'spatial_or_native_or_training_qualified':False,'planned':[dict(j,status='QUEUED') for j in p['required_jobs']]+[dict(p['conditional_job'],status='CONDITIONAL_QUEUED')]}
    def save():s['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',s)
    def interrupt(sig,frame):raise TimeoutError('Frozen300s/8GiB hard resource guard')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+300,8*1024**3,out/'resource-guard.json'),daemon=True);guard.start()
    fields={}
    for hz,r in p['six_tet_references'].items():
        with np.load(r['path'],allow_pickle=False) as a:fields[int(hz)]=a['probe_positions_m'].copy();points=a['material_probes_m'].copy()
    def capture(j):
        row=dict(j,completed=False);slot=next(x for x in s['planned'] if x['id']==j['id']);slot['status']='RUNNING';save();begin=time.monotonic();states=[];ledgers=[];commands=[]
        try:
            c=ConsistentP2Transient(tuple(j['cells']));m=c.model;state=c.initial();states=[state];prev=None;hz=j['hz'];steps=round(.1*hz);Q=query_operator(m,points)
            for n in range(steps):
                F=m.load*force_coefficient('smooth_ramp',(n+1)/hz);new,d=c.step(state,prev,1/hz,F);d.update(step=n+1,time_s=(n+1)/hz);ledgers.append(d);commands.append(F)
                if d['objective_free_R_N']>1e-10 or d['physical_original_free_R_N']>1e-9 or d['minimum_J']<=.1 or d['constraint_error_m']!=0 or d['constraint_velocity_error_m_s']!=0:raise ValueError('Original physical/constraint residual gate')
                prev,state=state,new;states.append(state)
            U=np.stack([a.u for a in states]);V=np.stack([a.v for a in states]);indices=np.arange(25)*(hz//240);queries=np.stack([points+Q@u for u in U[indices]]);M=c.mass;f=out/(j['id']+'.npz')
            np.savez_compressed(f,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,constraint_mask=m.mask,times_s=np.arange(steps+1)/hz,displacements_m=U,positions_m=m.rest+U,velocities_m_s=V,step_increments_m=np.stack([a.delta for a in states]),force_times_s=np.arange(1,steps+1)/hz,force_commands_N=np.stack(commands),unit_load_N=m.load,common_times_s=np.arange(25)/240,common_state_indices=indices,material_probes_m=points,probe_positions_m=queries,probe_velocities_m_s=np.stack([Q@v for v in V[indices]]),mass_csr_data=M.data,mass_csr_indices=M.indices,mass_csr_indptr=M.indptr,mass_shape=np.array(M.shape),mass_density_kg_m3=np.array(m.density))
            row.update(completed=True,steps=steps,nodes=len(m.rest),capture=record(f),maximum_objective_R_N=max(d['objective_free_R_N'] for d in ledgers),maximum_physical_R_N=max(d['physical_original_free_R_N'] for d in ledgers),minimum_J=min(d['minimum_J'] for d in ledgers),maximum_clamp_error_m=max(d['constraint_error_m'] for d in ledgers),maximum_clamp_velocity_m_s=max(d['constraint_velocity_error_m_s'] for d in ledgers));slot['status']='COMPLETED';return queries
        except Exception as e:
            row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED'
            if states:np.savez_compressed(out/(j['id']+'-failure-last-state.npz'),u=states[-1].u,v=states[-1].v,delta=states[-1].delta)
            raise
        finally:row['elapsed_s']=time.monotonic()-begin;s['cases'].append(row);dump(out/(j['id']+'.json'),row);dump(out/(j['id']+'-steps.json'),ledgers);save()
    try:
        for j in p['required_jobs']:fields[j['hz']]=capture(j)
        rates=[240,480,960,1920,3840];errors=[float(np.linalg.norm(fields[b]-fields[a],axis=-1).max()) for a,b in zip(rates[:-1],rates[1:])];s['six_tet_all_actual_rates_hz']=rates;s['six_tet_all_adjacent_query_error_m']=errors;s['six_tet_temporal_passed']=errors[-1]<=20e-6 and all(b<=a for a,b in zip(errors[:-1],errors[1:]));elapsed=time.monotonic()-start;forecast=p['larger_runtime_forecast_with10percent_safety_s']
        s['conditional_decision']={'elapsed_s':elapsed,'runtime_forecast_s':forecast,'eligible':eligible_larger_case(errors,elapsed,forecast)};save()
        if s['conditional_decision']['eligible']:
            q=capture(p['conditional_job'])
            with np.load(p['larger_reference480']['path'],allow_pickle=False) as a:delta=float(np.linalg.norm(q-a['probe_positions_m'],axis=-1).max())
            s['larger_480_to1920_max_query_difference_m']=delta;s['larger_480_to1920_within20um']=delta<=20e-6;s['larger_temporal_qualified']=False
        else:s['planned'][-1]['status']='SKIPPED_DECLARED_CONDITION';s['conditional_not_launched_reason']='6tet criterion or frozen remaining cost condition failed'
        s['completed']=True
    except Exception as e:s['execution_failure']=repr(e)
    finally:save();guard.terminate();guard.join(timeout=2)
    return s
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        s=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:s.get(k) for k in ['completed','elapsed_s','six_tet_all_adjacent_query_error_m','six_tet_temporal_passed','conditional_decision','larger_480_to1920_max_query_difference_m','execution_failure']},indent=2));raise SystemExit(0 if s['completed'] else 1)
