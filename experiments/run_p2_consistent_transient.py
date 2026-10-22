"""Bounded prospective ten-trajectory CPU consistent-P2 BDF2 reference."""
import argparse,json,multiprocessing,os,signal,time
from pathlib import Path
import numpy as np
from scipy import sparse
from experiments.campaign_p2_consistent_transient import ConsistentP2Transient,force_coefficient
from experiments.campaign_static_p2_block import shapes
from experiments.diagnose_nonlinear_physical_probes import interpolate
from experiments.run_static_supported_block import probes
from experiments.run_static_p2_axial_continuation import sha,dump,record,watchdog


def jobs():
    return [{'id':f'tets{tet}-{program}-hz{hz}','cells':cells,'tetrahedra':tet,'program':program,'hz':hz} for tet,cells,rates in [(6,[1,1,1],[240,480,960]),(96,[4,2,2],[480,960])] for program in ['step','smooth_ramp'] for hz in rates]


def prepare(path):
    sources=['experiments/run_p2_consistent_transient.py','experiments/campaign_p2_consistent_transient.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/campaign_stable_transient.py','experiments/run_static_supported_block.py','experiments/run_static_p2_axial_continuation.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    p={'schema':'C38-consistent-P2-transient-protocol-v0.1','status':'FROZEN_BEFORE_TRANSIENT_TRAJECTORIES','source_sha256':{x:sha(x) for x in sources},'CPU_threads':2,'hard_wall_cap_s':300,'actual_RSS_guard_bytes':8*1024**3,'forecast_s':[30,120],'jobs':jobs(),'total_steps':624,'duration_s':.1,'common_output_hz':240,'quadrature_order':3,'physics':{'size_m':[.24,.08,.08],'E_Pa':5000.,'nu':.35,'rho_kg_m3':625.,'gravity':0.,'damping':0.,'contact':'UNMODELED','boundary':'Full left face fixed all axes','load':'1 N normalized full right-face axial dead traction','material':'Newton stable NH, kappa=lambda+mu','mass':'Exact analytic T10 consistent sparse scalar matrix; full node coupling, no row lumping'},'method':{'startup':'Backward Euler','thereafter':'BDF2 on first-order (u,v)','effective_dt':'2*h/3','increment_predictor':'delta_previous/3+(2*h/9)*(4*v_current-v_previous)','velocity':'(3*delta_new-delta_previous)/(2*h)'},'command_clock':{'evaluation':'RIGHT_ENDPOINT t=(n+1)/physics_hz','step':1.,'smooth_ramp':'sin²((pi/2)*min(t/0.05,1)); then hold1','integer_steps':'round(duration_s*hz)','not_final_neural_actuator_clock':True},'gates':{'original_objective_R_N':1e-10,'physical_original_R_N':1e-9,'minimum_sampled_J':.1,'exact_clamp_position_m':0.,'exact_clamp_velocity_m_s':0.,'temporal_fixed_query_max_m':20e-6},'limits':['Fixed discrete meshes only; no spatial qualification','All full trajectories retained, no draw or force adaptation','Two histories and all ten cases predefined; no automatic extra rates','Continuous J certificate not claimed by transient sampled guard','No native GPU, contact, inference or training qualification']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,p);return p


def query_operator(m,points):
    _,e,b=interpolate(m.rest[:m.tets.max()+1],m.tets,m.rest[None,:m.tets.max()+1],points)
    return sparse.coo_matrix((shapes(b).ravel(),(np.repeat(np.arange(len(points)),10),m.elements[e].ravel())),shape=(len(points),len(m.rest))).tocsr()


def run(path,digest,output):
    if sha(path)!=digest:raise ValueError('Changed frozen protocol')
    p=json.loads(Path(path).read_text())
    if p['jobs']!=jobs():raise ValueError('Changed exact trajectory matrix')
    for source,h in p['source_sha256'].items():
        if sha(source)!=h:raise ValueError('Changed source '+source)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        if os.environ.get(name)!='2':raise ValueError('Explicit two CPU threads required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'schema':'C38-consistent-P2-transient-result-v0.1','completed':False,'protocol':record(path),'cases':[],'planned':[dict(x,status='QUEUED') for x in jobs()],'comparisons':[],'fixed_mesh_temporal_qualified':False,'spatial_or_native_or_training_qualified':False}
    def save():result['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',result)
    def interrupt(s,f):raise TimeoutError('External 300s/8GiB hard resource guard')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+300,8*1024**3,out/'resource-guard.json'),daemon=True);guard.start()
    fields={}
    try:
        points=np.concatenate(list(probes().values()))
        for job in p['jobs']:
            slot=next(x for x in result['planned'] if x['id']==job['id']);slot['status']='RUNNING';save();begin=time.monotonic();row=dict(job,completed=False);states=[];ledgers=[];forces=[]
            try:
                c=ConsistentP2Transient(tuple(job['cells']),order=3);m=c.model;Q=query_operator(m,points);state=c.initial();previous=None;states=[state];hz=job['hz'];steps=round(.1*hz);h=1/hz
                for n in range(steps):
                    force=m.load*force_coefficient(job['program'],(n+1)/hz)
                    new,d=c.step(state,previous,h,force);d['step']=n+1;d['time_s']=(n+1)/hz
                    ledgers.append(d);forces.append(force)
                    if d['objective_free_R_N']>1e-10 or d['physical_original_free_R_N']>1e-9 or d['minimum_J']<=.1 or d['constraint_error_m']!=0 or d['constraint_velocity_error_m_s']!=0:raise ValueError('Original strict physical/constraint gate failed')
                    previous,state=state,new;states.append(state)
                U=np.stack([s.u for s in states]);V=np.stack([s.v for s in states]);delta=np.stack([s.delta for s in states]);indices=np.arange(25)*(hz//240);q=np.stack([points+Q@u for u in U[indices]]);v=np.stack([Q@a for a in V[indices]])
                capture=out/(job['id']+'.npz');M=c.mass
                np.savez_compressed(capture,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,constraint_mask=m.mask,times_s=np.arange(steps+1)/hz,displacements_m=U,positions_m=m.rest+U,velocities_m_s=V,step_increments_m=delta,force_times_s=np.arange(1,steps+1)/hz,force_commands_N=np.stack(forces),unit_load_N=m.load,common_times_s=np.arange(25)/240,common_state_indices=indices,material_probes_m=points,probe_positions_m=q,probe_velocities_m_s=v,mass_csr_data=M.data,mass_csr_indices=M.indices,mass_csr_indptr=M.indptr,mass_shape=np.array(M.shape),mass_density_kg_m3=np.array(m.density))
                row.update(completed=True,steps=steps,nodes=len(m.rest),maximum_objective_R_N=max(x['objective_free_R_N'] for x in ledgers),maximum_physical_original_R_N=max(x['physical_original_free_R_N'] for x in ledgers),minimum_J=min(x['minimum_J'] for x in ledgers),maximum_constraint_error_m=max(x['constraint_error_m'] for x in ledgers),maximum_constraint_velocity_error_m_s=max(x['constraint_velocity_error_m_s'] for x in ledgers),maximum_newton_iterations=max(x['iterations'] for x in ledgers),mass_total_kg=float(M.sum()),capture=record(capture));slot['status']='COMPLETED';fields[(job['tetrahedra'],job['program'],hz)]=q
            except Exception as e:
                row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED'
                if states:
                    np.savez_compressed(out/(job['id']+'-failure-last-state.npz'),u=states[-1].u,v=states[-1].v,delta=states[-1].delta)
                raise
            finally:
                row['elapsed_s']=time.monotonic()-begin;result['cases'].append(row);dump(out/(job['id']+'.json'),row);dump(out/(job['id']+'-steps.json'),ledgers);save()
        for tet in [6,96]:
            for program in ['step','smooth_ramp']:
                rates=[240,480,960] if tet==6 else[480,960];errors=[float(np.linalg.norm(fields[(tet,program,b)]-fields[(tet,program,a)],axis=-1).max()) for a,b in zip(rates[:-1],rates[1:])]
                result['comparisons'].append({'tetrahedra':tet,'program':program,'rates_hz':rates,'adjacent_max_query_error_m':errors,'finest_adjacent_passed':errors[-1]<=20e-6,'contraction_test_available':len(errors)>1,'monotonic_adjacent_errors':all(b<=a for a,b in zip(errors[:-1],errors[1:])) if len(errors)>1 else None})
        result['completed']=True;result['fixed_mesh_temporal_qualified']=all(x['finest_adjacent_passed'] and x['monotonic_adjacent_errors'] is not False for x in result['comparisons'])
    except Exception as e:result['execution_failure']=repr(e)
    finally:save();guard.terminate();guard.join(timeout=2)
    return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        r=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:r.get(k) for k in ['completed','elapsed_s','fixed_mesh_temporal_qualified','comparisons','execution_failure']},indent=2));raise SystemExit(0 if r['completed'] else 1)
