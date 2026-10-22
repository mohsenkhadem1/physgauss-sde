"""C38 same stableNH law P2 spatial refinement; oldC36/C37remainimmutable."""
import argparse,json,os,signal,time,multiprocessing
from pathlib import Path
import numpy as np
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.run_static_p2_axial_continuation import sha,dump,record,checked_record,comparison,watchdog
BASE=Path('runs/gaussian-campaign-v01/newton-material-alignment-v01')

def admission(rows,spatial,quadrature):
    complete=len(rows)==2 and all(r.get('completed',False) for r in rows)
    physics=complete and all(r.get('physics_passed',False) for r in rows)
    positive=complete and all(r.get('global_affine_F_positive_J_sufficient_lower_bound',0) is not None and r.get('global_affine_F_positive_J_sufficient_lower_bound',0)>0 for r in rows)
    excitation=complete and all(r.get('excitation_goal_met',False) for r in rows)
    spatial_ok=bool(spatial.get('passed',False));quad_ok=bool(quadrature.get('passed',False))
    return dict(finest_physics_passed=physics,positive_global_J_certificate_passed=positive,spatial_indicator_passed=spatial_ok,quadrature_passed=quad_ok,excitation_goal_met=excitation,static_fixed_probe_qualified=physics and positive and spatial_ok and quad_ok,integrated_static_case_passed=physics and positive and spatial_ok and quad_ok and excitation)

def prepare(path):
    old=json.loads((BASE/'summary.json').read_text())
    if not old['completed']:raise ValueError('C37stablelawstageincomplete')
    rows=[r for r in old['cases'] if r['basis']=='P2' and r['tetrahedra']==6144]
    if len(rows)!=2 or not all(r['completed'] for r in rows):raise ValueError('Exact6144stablelawreference missing')
    r=next(r for r in rows if r['order']==3);forecast=sum(r['elapsed_s'] for r in rows)*(12000/6144)**1.7
    if 1.1*forecast>=900:raise ValueError('Prospective900sbudgetforecast exceeded')
    sources=['experiments/run_newton_stable_p2_spatial.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/run_static_p2_axial_continuation.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    p={'schema':'C38-stableNH-P2-spatial-protocol-v0.1','status':'FROZEN_BEFORE_FINE_STABLELAW_FIELDS','old_stable_summary':record(BASE/'summary.json'),'old_stable_protocol':old['protocol'],'reference6144':r['capture'],'source_sha256':{f:sha(f) for f in sources},'CPU_threads':2,'hard_wall_cap_s':900,'actual_RSS_guard_bytes':8*1024**3,'runtime_forecast_s':forecast,'forecast_with_safety_s':1.1*forecast,'exact_cells':[20,10,10],'tetrahedra':12000,'quadrature_orders':[3,4],'all_from_REST':True,'physics':{'E_Pa':5000.,'nu':.35,'rho_kg_m3':625.,'size_m':[.24,.08,.08],'traction_N':1.,'gravity':0.,'boundary':'Fullleftfaceall3axes','law':'Newton1.7stableNHkappa=lambdaL+muL'},'gates':{'original_free_R_N':1e-11,'max_query_delta_m':20e-6,'conditional_p_ge1_fine_error_indicator':'4*delta6144_to12000<=20um','quadrature_delta_m':20e-6,'minimum_sampled_J':.1,'positive_global_J_certificate':True,'exact_clamp_m':0.,'force_balance_N':1e-8,'actual_moment_balance_Nm':1e-8,'sampled_max_absolute_principal_strain':[.02,.08],'nonrigid_volume_coordinate_RMS_min_m':.001},'limits':['Conditionalp>=1engineeringindicatorNOTprovencontinuumerrorbound','NochangeoldlogJorstablelawfailures','Staticexactcaseonly;nodynamic/training/inference/nativeadmission','BothquadraturesfromREST;nonewlevelsorforcechanges']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,p)
def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(protocol).read_text());checked_record(p['old_stable_summary']);checked_record(p['old_stable_protocol'])
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    r=p['reference6144']
    if sha(r['path'])!=r['sha256']:raise ValueError('Changed stablelaw reference')
    with np.load(r['path'],allow_pickle=False) as a:reference=a['probe_positions_m'].copy();points=a['material_probes_m'].copy()
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();d={'schema':'C38-stableNH-P2-spatial-result-v0.1','completed':False,'protocol':record(protocol),'cases':[],'comparisons':[],'planned':[{'order':o,'status':'QUEUED'} for o in [3,4]],'integrated_static_case_passed':False,'training_or_dynamic_or_inference_qualified':False,'old_failed_gates_preserved':True};fields={}
    def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
    def interrupted(s,f):raise TimeoutError('External900s/8GiBresourceguard')
    signal.signal(signal.SIGTERM,interrupted);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+900,8*1024**3,out/'resource-guard.json'),daemon=True);guard.start()
    try:
        for order in [3,4]:
            row={'basis':'P2','cells':[20,10,10],'tetrahedra':12000,'order':order,'completed':False};begin=time.monotonic();slot=next(r for r in d['planned'] if r['order']==order);slot['status']='RUNNING';save()
            try:
                m=StableP2Block((20,10,10),young=5000.,poisson=.35,density=625.,gravity=0.,traction=1.,order=order);a=m.solve();row.update(m.diagnostics(a));row['nodes']=len(m.rest);row['completed']=True;row['history']=a['history'];q=m.query(a['positions'],points)
                row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=1e-11 and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
                row['excitation_goal_met']=.02<=row['maximum_absolute_principal_stretch_strain']<=.08 and row['bulk_nonrigid_coordinate_RMS_m']>=.001
                path=out/f'stable-P2-tets12000-order{order}.npz';np.savez_compressed(path,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=a['positions'],material_probes_m=points,probe_positions_m=q,applied_forces_N=m.load,constraint_mask=m.mask);row['capture']=record(path);fields[order]=q;slot['status']='COMPLETED'
            except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED';raise
            finally:row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/f'order{order}.json',row);save()
        spatial=comparison(reference,fields[3],4);spatial['kind']='spatial6144_to12000';quad=comparison(fields[3],fields[4],1);quad['kind']='quadrature12000';d['comparisons']=[spatial,quad];d.update(admission(d['cases'],spatial,quad));d['completed']=True
    except Exception as e:d['execution_failure']=repr(e)
    finally:save();guard.terminate();guard.join(timeout=2)
    return d
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','integrated_static_case_passed','comparisons','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
