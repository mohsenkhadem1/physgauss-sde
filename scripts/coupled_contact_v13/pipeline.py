"""Separate reference generation, hypothesis library and sealed cross-interaction evaluation."""
from pathlib import Path
import sys,os,json,argparse,hashlib,time,traceback,concurrent.futures,multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scripts.coupled_contact_v13.campaign import BASE,REPORT,ROOT,write,remaining
from scripts.coupled_contact_v13.scenarios import build,run,command,shared_node_indices
from scripts.coupled_contact_v13.inference import calibrate,observation_nodes
from scripts.coupled_contact_v13.development import shape_error

def protocol():return json.loads((REPORT/'protocol.json').read_text())
def check_freeze():
    d=json.loads((REPORT/'method-freeze.json').read_text())
    for p,h in d['source_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+p)
    if hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()!=d['protocol_sha256']:raise RuntimeError('Protocol changed')

def model_class(family):
    if family in ('fem','oracle'):
        from scripts.coupled_contact_v13.physics import FEMSystem
        return FEMSystem
    if family=='native':
        from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem
        return NativeSpringSystem
    if family=='rigid':
        from scripts.coupled_contact_v13.rigid import RigidSystem
        return RigidSystem
    raise ValueError(family)

def save_result(path,result):
    path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    write(path/'diagnostics.json',{k:v for k,v in result.items() if isinstance(v,(float,int,bool,str))})

def simulate_job(job):
    p=protocol();path=BASE/job['path'];path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return json.loads((path/'complete.json').read_text())
    start=time.monotonic()
    try:
        if remaining()<=0:raise RuntimeError('Campaign deadline reached')
        result=run(job['scenario'],job['scale'],job['friction'],tuple(p['reference_cells'] if job['reference'] else p['coarse_cells']),
                   (p['dt_reference']*(.5 if job.get('refinement',False) else 1)) if job['reference'] else p['dt_coarse'],p['duration_seconds'],model_class(job['family']))
        save_result(path,result)
        status=dict(success=True,job=job,seconds=time.monotonic()-start,diagnostics=json.loads((path/'diagnostics.json').read_text()))
    except Exception as e:
        status=dict(success=False,job=job,seconds=time.monotonic()-start,error=repr(e),traceback=traceback.format_exc())
    write(path/'complete.json',status);return status

def execute(jobs,workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[]
        for row in pool.map(simulate_job,jobs):
            rows.append(row);print(json.dumps({k:row[k] for k in ('success','job','seconds')}),flush=True)
    return rows

def library_jobs():
    p=protocol();jobs=[]
    for family,scales in [('fem',p['fem_young_grid']),('native',p['native_scale_grid']),('rigid',[6000.])]:
        for i,scale in enumerate(scales):
            for j,mu in enumerate(p['friction_grid']):jobs.append(dict(path=f'library/{family}/{i}-{j}',family=family,scenario='squeeze',scale=scale,friction=mu,reference=False))
    return jobs

def reference_jobs(split):
    p=protocol();jobs=[dict(path=f'reference/{o["id"]}/{s}',family='fem',scenario=s,scale=o['young'],friction=o['friction'],reference=True)
        for o in p['objects'] if o['split']==split for s in ['squeeze']+p['forecast_scenarios']]
    return jobs+[dict(j,path=j['path'].replace('reference/','reference-refinement/'),refinement=True) for j in jobs]

def load_libraries():
    libs={k:[] for k in ['fem','native','rigid']}
    for job in library_jobs():
        path=BASE/job['path'];status=json.loads((path/'complete.json').read_text());r=dict(scale=job['scale'],friction=job['friction'],frames=None)
        if status['success']:
            with np.load(path/'trajectory.npz') as d:r['frames']=d['frames']
        libs[job['family']].append(r)
    return libs

def prepare_object(o):
    p=protocol();path=BASE/'reference'/o['id']/'squeeze'
    status=json.loads((path/'complete.json').read_text())
    if not status['success']:return None,None,None
    coarse=build('squeeze',cells=tuple(p['coarse_cells']))
    with np.load(path/'trajectory.npz') as d:
        ids=shared_node_indices(d['rest'],coarse.rest);observed=d['frames'][:,ids].copy()
        tools=d['tool_positions'].copy();times=d['times'].copy();active=d['active'].copy()
    nodes=observation_nodes(coarse.rest,p['observation_nodes']);mask=np.zeros(observed.shape[:2],bool);mask[:,nodes]=True
    seed=p['observation_seed']+next(i for i,x in enumerate(p['objects']) if x['id']==o['id'])
    observed+=np.random.default_rng(seed).normal(0,p['observation_noise_std_m'],observed.shape);observed[0]=coarse.rest
    observed[~mask]=np.nan
    selected,empirical=calibrate(coarse.rest,observed,mask,tools,times,active,load_libraries())
    save=BASE/'inference'/o['id'];save.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(save/'observations.npz',rest=coarse.rest,observed=observed,mask=mask,tools=tools,times=times,active=active)
    write(save/'selection.json',selected)
    return selected,empirical,coarse

def metrics(pred,target):
    if not np.isfinite(target).all():return dict(complete=False,coordinate_rmse_m=None,shape_rmse_m=None,reason='Nonfinite reference target')
    if pred is None or pred.shape!=target.shape or not np.isfinite(pred).all():return dict(complete=False,coordinate_rmse_m=None,shape_rmse_m=None)
    return dict(complete=True,coordinate_rmse_m=float(np.sqrt(np.mean((pred[1:]-target[1:])**2))),
                shape_rmse_m=float(np.sqrt(np.mean([shape_error(a,b)**2 for a,b in zip(pred[1:],target[1:])]))))

def reference_quality(object_id,scenario):
    p=protocol();paths=[BASE/'reference'/object_id/scenario,BASE/'reference-refinement'/object_id/scenario]
    reasons=[];arrays=[]
    for label,path in zip(['base','half-dt'],paths):
        try:
            status=json.loads((path/'complete.json').read_text())
            if not status['success']:reasons.append(label+' generation failed');continue
            d=status['diagnostics']
            if not np.isfinite(d['max_penetration_m']) or d['max_penetration_m']>p['gates']['penetration_m']:reasons.append(label+' penetration exceeds limit')
            if not np.isfinite(d['min_j']) or d['min_j']<=.02:reasons.append(label+' invalid element volume')
            with np.load(path/'trajectory.npz') as a:
                frames=a['frames'].copy();times=a['times'].copy()
            if not np.isfinite(frames).all() or not np.isfinite(times).all():reasons.append(label+' nonfinite reference')
            arrays.append((frames,times))
        except (OSError,ValueError,KeyError) as e:reasons.append(label+' reference unavailable: '+str(e))
    rms=None
    if len(arrays)==2:
        (a,ta),(b,tb)=arrays
        if a.shape!=b.shape or ta.shape!=tb.shape or not np.allclose(ta,tb,rtol=0,atol=1e-10):reasons.append('Reference refinement grid mismatch')
        else:
            rms=float(np.sqrt(np.mean((a-b)**2)))
            if not np.isfinite(rms) or rms>p['gates']['trajectory_refinement_rms_m']:reasons.append('Reference timestep difference exceeds limit')
    else:reasons.append('Two reference levels unavailable')
    return dict(qualified=not reasons,reasons=reasons,rms_m=rms,interpretation='Per-case two-level absolute consistency; three-level refinement trend qualified on development only')

def evaluate(split,workers):
    p=protocol();objects=[o for o in p['objects'] if o['split']==split];models={};jobs=[];unavailable=[]
    for o in objects:
        selected,empirical,_=prepare_object(o);models[o['id']]=(selected,empirical)
        if selected is None:unavailable.append(o['id']);continue
        for s in p['forecast_scenarios']:
            for family,r in selected.items():
                if r['success']:jobs.append(dict(path=f'prediction/{o["id"]}/{s}/{family}',family=family,scenario=s,scale=r['scale'],friction=r['friction'],reference=False))
            jobs.append(dict(path=f'prediction/{o["id"]}/{s}/oracle',family='oracle',scenario=s,scale=o['young'],friction=o['friction'],reference=False))
    execute(jobs,workers);rows=[]
    for o in objects:
        selected,empirical=models[o['id']]
        for s in p['forecast_scenarios']:
            path=BASE/'reference'/o['id']/s;status=json.loads((path/'complete.json').read_text());row=dict(object=o['id'],split=split,scenario=s,reference_generation_success=status['success'],reference_qualified=False,reference_qualification_reasons=[],results={})
            forecast_quality=reference_quality(o['id'],s)
            calibration_quality=reference_quality(o['id'],'squeeze')
            row.update(reference_qualified=forecast_quality['qualified'],reference_qualification_reasons=forecast_quality['reasons'],
                       reference_refinement_rms_m=forecast_quality.get('rms_m'),calibration_reference_qualified=calibration_quality['qualified'],
                       calibration_qualification_reasons=calibration_quality['reasons'],
                       joint_reference_qualified=forecast_quality['qualified'] and calibration_quality['qualified'])
            if not status['success'] or empirical is None:
                row['unavailable_reason']='reference/calibration generation failed';rows.append(row);continue
            coarse=build(s,cells=tuple(p['coarse_cells']))
            with np.load(path/'trajectory.npz') as d:
                ids=shared_node_indices(d['rest'],coarse.rest);target=d['frames'][:,ids].copy();body=d['centers'].copy();times=d['times'].copy()
                tools=d['tool_positions'].copy();active=d['active'].copy()
            # Empirical methods receive only public tools/time/initial geometry.
            predictions=empirical.predict(coarse.rest,tools,times,active)
            for family in ['fem','native','rigid','oracle']:
                q=BASE/'prediction'/o['id']/s/family;out=json.loads((q/'complete.json').read_text()) if (q/'complete.json').exists() else {'success':False}
                pred=None
                if out['success']:
                    with np.load(q/'trajectory.npz') as d:
                        pred=d['frames'].copy();bodypred=d['centers'].copy()
                    row['results'][family]=metrics(pred,target);row['results'][family]['diagnostics']=out['diagnostics']
                    if s=='moving_sphere':row['results'][family]['dynamic_body_position_rmse_m']=float(np.sqrt(np.mean(np.sum((bodypred[1:,-1]-body[1:,-1])**2,axis=1))))
                else:row['results'][family]=metrics(None,target);row['results'][family]['failure']=out.get('error','No valid calibration hypothesis')
                if pred is not None:predictions[family]=pred
            for family in ['persistence','position_ridge','position_rigid']:row['results'][family]=metrics(predictions[family],target)
            save=BASE/'evaluation'/o['id']/s;save.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(save/'predictions.npz',**predictions)
            write(save/'metrics.json',row);rows.append(row)
    report=dict(split=split,expected_cases=len(objects)*len(p['forecast_scenarios']),cases=rows,calibration_unavailable=unavailable,
                selections={o['id']:models[o['id']][0] for o in objects},scope=p['scope'])
    write(REPORT/f'{split}-results.json',report);return report

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['library','reference','evaluate']);a.add_argument('--split',choices=['development','validation','reserved'],default='development');a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    check_freeze()
    if args.stage=='library':write(REPORT/'library-status.json',execute(library_jobs(),args.workers))
    elif args.stage=='reference':write(REPORT/f'{args.split}-reference-status.json',execute(reference_jobs(args.split),args.workers))
    else:evaluate(args.split,args.workers)
if __name__=='__main__':main()
