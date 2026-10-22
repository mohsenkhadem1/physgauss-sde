"""Independent bounded spatial continuation, no edits/relabeling of P2v01."""
import argparse,hashlib,json,os,signal,time,multiprocessing
from pathlib import Path
import numpy as np
from experiments.campaign_static_p2_block import QuadraticBlock

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def comparison(a,b,multiplier):
    delta=np.linalg.norm(np.asarray(a)-np.asarray(b),axis=1)
    return {'maximum_Euclidean_error_m':float(delta.max()),'minimum_order_one_fine_error_estimate_m':float(multiplier*delta.max()),'multiplier':multiplier,'passed':bool(delta.max()<=20e-6 and multiplier*delta.max()<=20e-6)}
def checked_record(r):
    if sha(r['path'])!=r['sha256']:raise ValueError('Changed evidence '+r['path'])
    return json.loads(Path(r['path']).read_text())
def record(p):return {'path':str(p),'sha256':sha(p)}
def prepare(path):
    base=Path('runs/gaussian-campaign-v01/static-p2-block-v01');old=json.loads((base/'summary.json').read_text())
    if not old['completed']:raise ValueError('Original stage must be terminal')
    rows=[r for r in old['cases'] if r['case_id']=='left_broad_axial'];r3=next(r for r in rows if r['tetrahedra']==6144 and r['order']==3);r4=next(r for r in rows if r['tetrahedra']==6144 and r['order']==4)
    if not r3['completed'] or not r4['completed']:raise ValueError('Fine reference captures missing')
    # Measured768→6144 scaling is superlinear; use at least1.7 power and
    # retain budget-infeasible optional work as unlaunched, never optimistic.
    small=next(r for r in rows if r['tetrahedra']==768 and r['order']==3);power=max(1.7,float(np.log(r3['elapsed_s']/small['elapsed_s'])/np.log(8)))
    forecast2592=r3['elapsed_s']*(2592/6144)**power
    forecast12000=(r3['elapsed_s']+r4['elapsed_s'])*(12000/6144)**power
    sources=['experiments/run_static_p2_axial_continuation.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
    p={'schema':'P2-axial-continuation-protocol-v0.1','status':'FROZEN_BEFORE_NEW_SOLVE','old_summary':record(base/'summary.json'),'old_protocol':old['protocol'],'old_axial_captures':[r['capture'] for r in rows],'source_sha256':{s:sha(s) for s in sources},'CPU_threads':2,'hard_wall_cap_s':600,'actual_RSS_guard_bytes':8*1024**3,'cases':[{'cells':[12,6,6],'order':3,'physics':'unchanged leftclamp E5000 nu.35 rho625,1Nuniformrighttraction,gravity0','initial':'REST'}],'gate':{'fixed_query_threshold_m':20e-6,'12_to16_fine_error_multiplier':3,'16_to20_fine_error_multiplier':4,'physical_force_and_constraint_gates':'unchanged P2v01','excitation':'unchanged 2–8%maxstrain and1mm nonrigidvolumeRMS','quadrature':'6144Q3/Q4 reused;12000Q3/Q4 if required and budget-feasible'},'conditional_single_finer_level':{'cells':[20,10,10],'orders':[3,4],'only_if_2592_to6144_gate_fails':True,'only_if_forecast_plus_elapsed_and_10percent_safety_within600s':True,'no_more_levels':True},'runtime_forecast':{'measured6144_Q3_s':r3['elapsed_s'],'measured6144_Q4_s':r4['elapsed_s'],'power':power,'2592_s':forecast2592,'optional12000_pair_s':forecast12000,'interpretation':'Prospective cost guard; retain unlaunched optional work if infeasible'},'preserve':'Original22.353umfailed comparison and gravityfailure immutable; no loads/supports/gates tuned'}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,p)

def watchdog(pid,deadline,limit,path):
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    peak=0
    while Path('/proc/'+str(pid)).exists():
        try:rss=int(Path('/proc/'+str(pid)+'/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError:return
        peak=max(peak,rss);reason='RSS_LIMIT' if rss>limit else('WALL_LIMIT' if time.monotonic()>deadline else None)
        dump(path,{'observed_peak_RSS_bytes':peak,'reason':reason})
        if reason:
            os.kill(pid,signal.SIGTERM);time.sleep(2)
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            return
        time.sleep(.5)

def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(protocol).read_text());old=checked_record(p['old_summary']);checked_record(p['old_protocol'])
    for r in p['old_axial_captures']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed original capture')
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();d={'schema':'P2-axial-continuation-result-v0.1','completed':False,'protocol':record(protocol),'new_cases':[],'comparisons':[],'static_fixed_probe_qualified':False,'original_failed_gate_preserved':True,'dynamic_or_inference_qualified':False}
    def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
    def terminated(s,f):raise TimeoutError('External hard RSS/time guard triggered')
    signal.signal(signal.SIGTERM,terminated);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+600,8*1024**3,out/'resource-guard.json'),daemon=True);guard.start()
    rows={};fields={}
    for r in old['cases']:
        if r['case_id']=='left_broad_axial' and r['completed']:
            with np.load(r['capture']['path'],allow_pickle=False) as a:fields[(r['tetrahedra'],r['order'])]=a['probe_positions_m'].copy();points=a['material_probes_m'].copy()
            rows[(r['tetrahedra'],r['order'])]=r
    def capture(cells,order):
        row={'cells':list(cells),'tetrahedra':int(6*np.prod(cells)),'order':order,'completed':False};begin=time.monotonic();label='axial-tets'+str(row['tetrahedra'])+'-order'+str(order)
        try:
            m=QuadraticBlock(cells,young=5000.,gravity=0.,traction=1.,order=order);a=m.solve();row.update(m.diagnostics(a));row['completed']=True;row['history']=a['history'];field=m.query(a['positions'],points)
            row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=1e-11 and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
            row['excitation_goal_met']=.02<=row['maximum_absolute_principal_stretch_strain']<=.08 and row['bulk_nonrigid_coordinate_RMS_m']>=.001
            file=out/(label+'.npz');np.savez_compressed(file,rest_m=m.rest,tets=m.tets,elements=m.elements,faces=m.faces,positions_m=a['positions'],probe_positions_m=field,material_probes_m=points);row['capture']=record(file);rows[(row['tetrahedra'],order)]=row;fields[(row['tetrahedra'],order)]=field
        except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};raise
        finally:row['elapsed_s']=time.monotonic()-begin;d['new_cases'].append(row);dump(out/(label+'.json'),row);save()
    try:
        capture((12,6,6),3)
        c=comparison(fields[(2592,3)],fields[(6144,3)],3);c.update(kind='spatial12_to16',coarse_tetrahedra=2592,fine_tetrahedra=6144);d['comparisons'].append(c)
        q=comparison(fields[(6144,3)],fields[(6144,4)],1);q.update(kind='quadrature6144');d['comparisons'].append(q)
        selected=6144
        if not c['passed']:
            forecast=p['runtime_forecast']['optional12000_pair_s']
            if time.monotonic()-start+1.1*forecast>600:
                d['optional_finer_status']='NOT_LAUNCHED_FORECAST_EXCEEDS_FROZEN_BUDGET';d['completed']=True;save();return d
            capture((20,10,10),3);capture((20,10,10),4);selected=12000
            c=comparison(fields[(6144,3)],fields[(12000,3)],4);c.update(kind='spatial16_to20',coarse_tetrahedra=6144,fine_tetrahedra=12000);d['comparisons'].append(c)
            q=comparison(fields[(12000,3)],fields[(12000,4)],1);q.update(kind='quadrature12000');d['comparisons'].append(q)
        physics=all(rows[(selected,o)]['physics_passed'] for o in [3,4]);excitation=all(rows[(selected,o)]['excitation_goal_met'] for o in [3,4])
        d.update(completed=True,selected_tetrahedra=selected,static_fixed_probe_qualified=c['passed'] and q['passed'] and physics,excitation_goal_met=excitation,finest_physics_passed=physics)
    except Exception as e:d['execution_failure']=repr(e)
    finally:save();guard.terminate();guard.join(timeout=2)
    return d
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(d,indent=2));raise SystemExit(0 if d['completed'] else 1)
