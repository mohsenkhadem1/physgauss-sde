"""Matched vertex-contact and fixed-nominal surface supplements, separate outputs."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,multiprocessing,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.surface_pipeline import implementation,BASE,REPORT
from scripts.coupled_contact_v13.engineered_scenarios import run,build,shared_node_indices
from scripts.coupled_contact_v13.physics import FEMSystem
from scripts.coupled_contact_v13.consistent_mass import with_consistent_mass
from scripts.coupled_contact_v13.surface_pipeline import simulation_settings
from scripts.coupled_contact_v13.inference import choose
from scripts.coupled_contact_v13.campaign import remaining,write

def read(path):return json.loads(path.read_text())
def load(path):
    with np.load(path) as a:return {k:a[k].copy() for k in a.files}

def simulate_job(job):
    m=implementation();p=m.protocol();path=BASE/job['path']
    settings=simulation_settings(dict(job,reference=False),p)
    if job['family']=='vertex':
        settings.update(contact_formulation='vertex-nodal-mass-damping',surface_spacing_m=None,friction_quadrature_level=None)
    protocol_sha=hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()
    if (path/'complete.json').exists():
        cached=read(path/'complete.json')
        if cached.get('job')!=job or cached.get('simulation_settings')!=settings or cached.get('protocol_sha256')!=protocol_sha:
            raise RuntimeError('Supplement cache provenance/settings mismatch: '+str(path))
        return cached
    start=time.monotonic()
    try:
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached')
        model=with_consistent_mass(FEMSystem) if job['family']=='vertex' else m.model_class('fem')
        result=run(job['scenario'],job['scale'],job['friction'],tuple(settings['cells']),settings['dt'],settings['duration_seconds'],model)
        m.save_result(path,result);status=dict(success=True,job=job,seconds=time.monotonic()-start,diagnostics=read(path/'diagnostics.json'))
    except Exception as error:status=dict(success=False,job=job,seconds=time.monotonic()-start,error=repr(error),traceback=traceback.format_exc())
    status.update(simulation_settings=settings,protocol_sha256=protocol_sha)
    write(path/'complete.json',status);return status

def execute(jobs,workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for status in pool.map(simulate_job,jobs):print(json.dumps({k:status[k] for k in ['success','job','seconds']}),flush=True)

def cohort_summary(rows,p):
    out={}
    for split in ('development','reserved','all_objects'):
        ids={o['id'] for o in p['objects'] if split=='all_objects' or o['split']==split};out[split]={}
        for name,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios']),('full',p['forecast_scenarios'])]:
            selected=[r for r in rows if r['object'] in ids and r['scenario'] in scenes];expected=len(ids)*len(scenes)
            record=dict(scenarios=scenes,expected=expected,present=len(selected),reference_qualified=sum(bool(r.get('joint_reference_qualified')) for r in selected),methods={})
            for method in ('vertex','nominal'):
                valid=[r['results'][method] for r in selected if r.get('results',{}).get(method,{}).get('complete')]
                gains=[r['surface_fem_gain_mm'][method] for r in selected if method in r.get('surface_fem_gain_mm',{})]
                record['methods'][method]=dict(expected=expected,complete=len(valid),failed_or_missing=expected-len(valid),means_conditional_on_success=len(valid)!=expected,mean_coordinate_mm=float(np.mean([r['coordinate_rmse_m'] for r in valid])*1000) if valid else None,paired_cases=len(gains),surface_fem_gain_mm=float(np.mean(gains)) if gains else None)
            out[split][name]=record
    return out

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--workers',type=int,default=4);args=arg.parse_args()
    m=implementation();m.check_freeze();p=m.protocol();supp=read(REPORT/'supplementary-protocol.json')
    for path,sha in supp['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Supplement frozen source changed: '+path)
    library=[dict(path=f'ablation/library/{i}-{j}',family='vertex',scenario='squeeze',scale=e,friction=mu) for i,e in enumerate(p['fem_young_grid']) for j,mu in enumerate(p['friction_grid'])]
    nominal=[dict(path=f'ablation/nominal/{s}',family='nominal',scenario=s,scale=6000.,friction=.35) for s in p['forecast_scenarios']]
    execute(library+nominal,args.workers)
    candidates=[]
    for job in library:
        path=BASE/job['path'];status=read(path/'complete.json')
        candidates.append(dict(scale=job['scale'],friction=job['friction'],frames=load(path/'trajectory.npz')['frames'] if status['success'] else None))
    selections={};jobs=[]
    for obj in p['objects']:
        observations=BASE/'inference'/obj['id']/'observations.npz'
        if not observations.exists():selections[obj['id']]=dict(success=False,reason='Calibration observations unavailable');continue
        a=load(observations);selection=choose(a['observed'],a['mask'],candidates);selections[obj['id']]=selection
        if selection['success']:
            jobs.extend(dict(path=f'ablation/prediction/{obj["id"]}/{s}',family='vertex',scenario=s,scale=selection['scale'],friction=selection['friction']) for s in p['forecast_scenarios'])
    write(BASE/'ablation/selections.json',selections);execute(jobs,args.workers);rows=[]
    main_rows={}
    for split in sorted({o['split'] for o in p['objects']}):
        main_rows.update({(r['object'],r['scenario']):r for r in read(REPORT/f'{split}-results.json')['cases']})
    for obj in p['objects']:
        for s in p['forecast_scenarios']:
            mainrow=main_rows[(obj['id'],s)];refpath=BASE/'reference'/obj['id']/s;refstatus=read(refpath/'complete.json')
            row=dict(object=obj['id'],split=obj['split'],scenario=s,reference_generation_success=refstatus['success'],joint_reference_qualified=mainrow['joint_reference_qualified'],results={})
            if not refstatus['success']:row['unavailable_reason']='Reference generation failed';rows.append(row);continue
            ref=load(refpath/'trajectory.npz');rest=build(s,cells=tuple(p['coarse_cells'])).rest;target=ref['frames'][:,shared_node_indices(ref['rest'],rest)]
            for family,path in [('vertex',BASE/'ablation/prediction'/obj['id']/s),('nominal',BASE/'ablation/nominal'/s)]:
                status=read(path/'complete.json') if (path/'complete.json').exists() else dict(success=False,error='No valid calibration hypothesis')
                a=load(path/'trajectory.npz') if status['success'] else None
                result=m.metrics(a['frames'] if a else None,target)
                if status['success']:
                    result['diagnostics']=status['diagnostics']
                    if s=='moving_sphere':result['dynamic_body_position_rmse_m']=float(np.sqrt(np.mean(np.sum((a['centers'][1:,-1]-ref['centers'][1:,-1])**2,axis=1))))
                else:result['failure']=status.get('error','Simulation failed')
                row['results'][family]=result
                fitted=mainrow['results'].get('fem',{})
                if result['complete'] and fitted.get('complete'):row.setdefault('surface_fem_gain_mm',{})[family]=(result['coordinate_rmse_m']-fitted['coordinate_rmse_m'])*1000
            rows.append(row)
    summary={}
    for split in sorted({o['split'] for o in p['objects']}):
        cohort=[r for r in rows if r['split']==split];summary[split]={}
        for method in ['vertex','nominal']:
            valid=[r['results'][method] for r in cohort if r['results'].get(method,{}).get('complete')];gain=np.array([r['surface_fem_gain_mm'][method] for r in cohort if method in r.get('surface_fem_gain_mm',{})])
            summary[split][method]=dict(expected=len(cohort),complete=len(valid),mean_coordinate_mm=float(np.mean([r['coordinate_rmse_m'] for r in valid])*1000) if valid else None,paired_cases=len(gain),surface_fem_gain_mm=float(gain.mean()) if len(gain) else None,wins=int((gain>1e-9).sum()),losses=int((gain< -1e-9).sum()),ties=int((abs(gain)<=1e-9).sum()))
    write(REPORT/'ablation-results.json',dict(expected_cases=len(p['objects'])*len(p['forecast_scenarios']),cases=rows,selections=selections,splits=summary,scope=supp['scope'],forecast_cohorts=cohort_summary(rows,p)))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
