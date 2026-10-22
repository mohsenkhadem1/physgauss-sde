"""Exactly one12000tet Q3/Q4 pair under preread900s resource amendment."""
import argparse,json,os,signal,time,multiprocessing
from pathlib import Path
import numpy as np
from experiments.campaign_static_p2_block import QuadraticBlock
from experiments.run_static_p2_axial_continuation import sha,dump,record,checked_record,comparison,watchdog
BASE=Path('runs/gaussian-campaign-v01/static-p2-block-v01')
CONT=Path('runs/gaussian-campaign-v01/static-p2-axial-continuation-v01')
AMEND=Path('runs/operator-dev/static-p2-axial-continuation-v01/budget-amendment-v02.json')
def prepare(path):
    old=json.loads((BASE/'summary.json').read_text());continuation=json.loads((CONT/'summary.json').read_text());a=json.loads(AMEND.read_text())
    if not old['completed'] or not continuation['completed'] or continuation['comparisons'][0]['passed']:raise ValueError('Exact continuation trigger absent')
    if a['hard_wall_cap_s']!=900 or sha(AMEND)!='7827a73c756cc397626440be32dfefae6eadc019426a6c45588649be76ff78b7':raise ValueError('Prospective amendment changed')
    pold=checked_record(continuation['protocol']);forecast=pold['runtime_forecast']['optional12000_pair_s']
    if 1.1*forecast>=900:raise ValueError('Frozen900s prospective cost guard failed')
    r=next(r for r in old['cases'] if r['case_id']=='left_broad_axial' and r['tetrahedra']==6144 and r['order']==3)
    sources=['experiments/run_static_p2_axial_fine.py','experiments/run_static_p2_axial_continuation.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    p={'schema':'P2-final-axial-pair-protocol-v0.1','status':'FROZEN_BEFORE12000_FIELDS','resource_amendment':record(AMEND),'old_summary':record(BASE/'summary.json'),'continuation_summary':record(CONT/'summary.json'),'reference6144':r['capture'],'source_sha256':{f:sha(f) for f in sources},'CPU_threads':2,'hard_wall_cap_s':900,'actual_RSS_guard_bytes':8*1024**3,'forecast_pair_s':forecast,'forecast_with_safety_s':1.1*forecast,'exact_cells':[20,10,10],'exact_orders':[3,4],'physics':'Identicalleftclamp,1Nbroadreferencefacetraction,E5000,nu.35,rho625,no gravity,fromREST','gates':{'max_query_delta_m':20e-6,'conservative_first_order_fine_error_estimate':'4*maxnorm(q12000_Q3-q6144_Q3)<=20um','quadrature':'maxnorm(q12000_Q4-q12000_Q3)<=20um','equilibrium_N':1e-11,'force_balance_N':1e-8,'moment_balance_Nm':1e-8,'minimum_sampled_J':.1,'exact_constraints_m':0.,'excitation':'2–8%max principal strain AND1mm rigidremovedvolumeRMS'},'no_additional_levels_or_force_changes':True,'limits':'Finiteprobe spatial development under minimumorder1 assumption; not certified continuum error, dynamics/native/inverse validation'}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,p)
def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(protocol).read_text())
    for name in ['resource_amendment','old_summary','continuation_summary']:checked_record(p[name])
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    r=p['reference6144']
    if sha(r['path'])!=r['sha256']:raise ValueError('Changed old capture')
    with np.load(r['path'],allow_pickle=False) as a:reference=a['probe_positions_m'].copy();points=a['material_probes_m'].copy()
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();d={'schema':'P2-final-axial-pair-result-v0.1','completed':False,'protocol':record(protocol),'cases':[],'planned':[{'order':o,'status':'QUEUED'} for o in [3,4]],'comparisons':[],'static_fixed_probe_qualified':False,'dynamic_or_inference_qualified':False,'all_original_failed_gates_preserved':True};fields={}
    def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
    def interrupted(s,f):raise TimeoutError('External900s/8GiB guard triggered')
    signal.signal(signal.SIGTERM,interrupted);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+900,8*1024**3,out/'resource-guard.json'),daemon=True);guard.start()
    try:
        for order in [3,4]:
            row={'tetrahedra':12000,'cells':[20,10,10],'order':order,'completed':False};begin=time.monotonic();slot=next(x for x in d['planned'] if x['order']==order);slot['status']='RUNNING';save()
            try:
                m=QuadraticBlock((20,10,10),young=5000.,gravity=0.,traction=1.,order=order);a=m.solve();row.update(m.diagnostics(a));row['nodes']=len(m.rest);row['completed']=True;row['history']=a['history'];field=m.query(a['positions'],points)
                row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=1e-11 and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
                row['excitation_goal_met']=.02<=row['maximum_absolute_principal_stretch_strain']<=.08 and row['bulk_nonrigid_coordinate_RMS_m']>=.001
                file=out/f'axial12000-order{order}.npz';np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=a['positions'],probe_positions_m=field,material_probes_m=points);row['capture']=record(file);fields[order]=field;slot['status']='COMPLETED'
            except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED';raise
            finally:row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/f'axial12000-order{order}.json',row);save()
        c=comparison(reference,fields[3],4);c['kind']='spatial16_to20';q=comparison(fields[3],fields[4],1);q['kind']='quadrature12000';d['comparisons']=[c,q];d['completed']=True;d['physics_passed']=all(r['physics_passed'] for r in d['cases']);d['excitation_goal_met']=all(r['excitation_goal_met'] for r in d['cases']);d['static_fixed_probe_qualified']=c['passed'] and q['passed'] and d['physics_passed']
    except Exception as e:d['execution_failure']=repr(e)
    finally:save();guard.terminate();guard.join(timeout=2)
    return d
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(d,indent=2));raise SystemExit(0 if d['completed'] else 1)
