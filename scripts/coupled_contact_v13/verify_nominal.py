"""Independent saved-array audit of the fixed-parameter supplementary baseline."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.verify import Audit,BASE,REPORT,read,arrays,digest,mapping,geometry_metrics,minimum_j

def main():
    audit=Audit();audit.hashes();p=read(REPORT/'protocol.json');supp=read(REPORT/'supplementary-nominal-protocol.json');report=read(REPORT/'nominal-baseline-results.json')
    for path,sha in supp['source_sha256'].items():audit.check(digest(ROOT/path)==sha,'supplement hash '+path)
    expected={(o['id'],s) for o in p['objects'] for s in p['forecast_scenarios']};present=[(r['object'],r['scenario']) for r in report['cases']]
    audit.check(len(present)==56 and len(set(present))==56 and set(present)==expected,'all56unique nominal cases')
    cohort={};nominal={}
    for scenario in p['forecast_scenarios']:
        q=BASE/'supplementary-nominal'/scenario;status=read(q/'complete.json');job=status['job']
        audit.check(job['scale']==6000. and job['friction']==.35 and job['family']=='fem' and not job['reference'],scenario+' fixed nominal parameters')
        if status['success']:
            a=arrays(q/'trajectory.npz');audit.close(status['diagnostics']['dt'],p['dt_coarse'],scenario+' nominal timestep')
            audit.check(minimum_j(a['frames'],a['rest'],a['tets'])>.02,scenario+' positive nominal saved volume');nominal[scenario]=a
    for row in report['cases']:
        obj,s,split=row['object'],row['scenario'],row['split'];label=obj+'/'+s
        if split not in cohort:
            cohort[split]={};main=read(REPORT/f'{split}-results.json')
            cohort[split]={(r['object'],r['scenario']):r for r in main['cases']}
        mainrow=cohort[split][(obj,s)];status=read(BASE/'reference'/obj/s/'complete.json')
        audit.check(row['reference_generation_success']==status['success'],label+' reference status')
        rq=audit.reference(obj,s,p);cq=audit.reference(obj,'squeeze',p)
        audit.check(row['joint_reference_qualified']==(rq['qualified'] and cq['qualified']),label+' joint qualification')
        if not status['success']:
            audit.check(not row['results'],label+' explicit unavailable reference');continue
        reference=arrays(BASE/'reference'/obj/s/'trajectory.npz');a=nominal.get(s);target=reference['frames'][:,mapping(reference['rest'],a['rest'])] if a else None
        calculated=geometry_metrics(a['frames'],target) if a else None;saved=row['results']['nominal_fem']
        audit.check(saved['complete']==(calculated is not None),label+' completeness')
        if calculated:
            for key,value in calculated.items():audit.close(saved[key],value,label+' '+key)
            if s=='moving_sphere':audit.close(saved['dynamic_body_position_rmse_m'],np.sqrt(np.mean(np.sum((a['centers'][1:,-1]-reference['centers'][1:,-1])**2,axis=1))),label+' body error')
            fitted=mainrow['results'].get('fem',{})
            if fitted.get('complete'):
                fittedarr=arrays(BASE/'evaluation'/obj/s/'predictions.npz')['fem'];independent=geometry_metrics(fittedarr,target)
                audit.close(row['inference_gain_mm'],(calculated['coordinate_rmse_m']-independent['coordinate_rmse_m'])*1000,label+' inference gain')
    result=dict(passed=not audit.errors,checks=audit.checks,errors=audit.errors,cases=len(present),verifier_sha256=digest(Path(__file__)))
    (REPORT/'nominal-baseline-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return int(not result['passed'])
if __name__=='__main__':sys.exit(main())
