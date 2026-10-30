"""Independent reconstruction of surface-study vertex and nominal supplements."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.surface_verify import implementation,verify_settings
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
independent=implementation()
Audit=independent.Audit
read,arrays,digest=independent.read,independent.arrays,independent.digest
mapping,geometry_metrics,minimum_j=independent.mapping,independent.geometry_metrics,independent.minimum_j

def main():
    audit=Audit();p=read(REPORT/'protocol.json');supp=read(REPORT/'supplementary-protocol.json');d=read(REPORT/'ablation-results.json');frozen=read(REPORT/'method-freeze.json')
    for path,sha in {**frozen['source_sha256'],**supp['source_sha256']}.items():audit.check(digest(ROOT/path)==sha,'frozen '+path)
    audit.check(digest(REPORT/'protocol.json')==frozen['protocol_sha256'],'main protocol unchanged')
    verify_settings(audit,p,'ablation')
    want={(o['id'],s) for o in p['objects'] for s in p['forecast_scenarios']};got=[(r['object'],r['scenario']) for r in d['cases']]
    audit.check(len(got)==40 and len(set(got))==40 and set(got)==want,'all40unique supplement cases')
    for obj in p['objects']:
        path=BASE/'inference'/obj['id']/'observations.npz';selected=d['selections'][obj['id']]
        if not path.exists():audit.check(not selected['success'],obj['id']+' missing calibration fails selection');continue
        a=arrays(path);losses=[];parameters=[]
        for i,e in enumerate(p['fem_young_grid']):
            for j,mu in enumerate(p['friction_grid']):
                q=BASE/'ablation/library'/f'{i}-{j}';status=read(q/'complete.json');job=status['job']
                audit.check(job['family']=='vertex' and job['scale']==e and job['friction']==mu and job['scenario']=='squeeze','matched recomputed vertex library '+str((i,j)))
                loss=np.inf
                if status['success']:
                    frames=arrays(q/'trajectory.npz')['frames']
                    if frames.shape==a['observed'].shape and np.isfinite(frames).all() and a['mask'].any():loss=float(np.mean((frames[a['mask']]-a['observed'][a['mask']])**2))
                losses.append(loss);parameters.append((e,mu))
        audit.check(np.allclose(losses,selected['scores_m2'],rtol=1e-10,atol=1e-16,equal_nan=True),obj['id']+' all35calibration losses')
        success=bool(np.isfinite(losses).any());audit.check(selected['success']==success,obj['id']+' calibration success')
        if success:
            index=int(np.argmin(losses));audit.check(selected['index']==index and (selected['scale'],selected['friction'])==parameters[index],obj['id']+' chosen vertex parameters')
    mainrows={}
    for split in sorted({o['split'] for o in p['objects']}):mainrows.update({(r['object'],r['scenario']):r for r in read(REPORT/f'{split}-results.json')['cases']})
    for row in d['cases']:
        obj,s=row['object'],row['scenario'];label=obj+'/'+s
        rq=audit.reference(obj,s,p);cq=audit.reference(obj,'squeeze',p);audit.check(row['joint_reference_qualified']==(rq['qualified'] and cq['qualified']),label+' joint reference quality')
        status=read(BASE/'reference'/obj/s/'complete.json');audit.check(row['reference_generation_success']==status['success'],label+' reference status')
        if not status['success']:audit.check(not row['results'],label+' unavailable reference retained');continue
        ref=arrays(BASE/'reference'/obj/s/'trajectory.npz')
        for family,q in [('vertex',BASE/'ablation/prediction'/obj/s),('nominal',BASE/'ablation/nominal'/s)]:
            status=read(q/'complete.json') if (q/'complete.json').exists() else dict(success=False)
            saved=row['results'][family]
            if not status['success']:audit.check(not saved['complete'],label+'/'+family+' failure accounting');continue
            arr=arrays(q/'trajectory.npz');job=status['job'];expected=(6000.,.35) if family=='nominal' else (d['selections'][obj]['scale'],d['selections'][obj]['friction'])
            audit.check((job['scale'],job['friction'])==expected and job['family']==family,label+'/'+family+' rollout parameters')
            audit.close(status['diagnostics']['dt'],p['dt_coarse'],label+'/'+family+' coarse timestep')
            audit.check(minimum_j(arr['frames'],arr['rest'],arr['tets'])>.02,label+'/'+family+' saved J')
            target=ref['frames'][:,mapping(ref['rest'],arr['rest'])];calc=geometry_metrics(arr['frames'],target)
            audit.check(saved['complete']==(calc is not None),label+'/'+family+' complete')
            if calc:
                for key,value in calc.items():audit.close(saved[key],value,label+'/'+family+' '+key)
                if s=='moving_sphere':audit.close(saved['dynamic_body_position_rmse_m'],np.sqrt(np.mean(np.sum((arr['centers'][1:,-1]-ref['centers'][1:,-1])**2,axis=1))),label+'/'+family+' body error')
                if mainrows[(obj,s)]['results'].get('fem',{}).get('complete'):
                    fitted=arrays(BASE/'evaluation'/obj/s/'predictions.npz')['fem'];score=geometry_metrics(fitted,target)
                    audit.close(row['surface_fem_gain_mm'][family],(calc['coordinate_rmse_m']-score['coordinate_rmse_m'])*1000,label+'/'+family+' paired gain')
    from scripts.coupled_contact_v13.surface_scope_audit import supplement_cohorts
    expected_cohorts=supplement_cohorts(d['cases'],p)
    def compare_tree(actual,expected,label):
        if isinstance(expected,dict):
            audit.check(isinstance(actual,dict) and set(actual)==set(expected),label+' keys')
            for key,value in expected.items():compare_tree(actual.get(key) if isinstance(actual,dict) else None,value,label+'/'+key)
        elif isinstance(expected,float):audit.close(actual,expected,label)
        else:audit.check(actual==expected,label)
    compare_tree(d.get('forecast_cohorts'),expected_cohorts,'Independent fixed supplement cohorts')
    result=dict(passed=not audit.errors,checks=audit.checks,errors=audit.errors,cases=len(got),audited_report_sha256={name:digest(REPORT/name) for name in ['protocol.json','method-freeze.json','ablation-results.json','supplementary-protocol.json']},verifier_sha256=digest(Path(__file__)))
    (REPORT/'ablation-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return int(not result['passed'])
if __name__=='__main__':sys.exit(main())
