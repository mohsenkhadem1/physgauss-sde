"""Read-only secondary grid comparison; fixed cohorts, failures never filtered."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
METHODS=('fem','native','rigid','oracle','position_ridge','position_rigid','persistence')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def average(v):return float(np.mean(v)) if v else None

def summarize(keys, fine, coarse):
    out={'expected':len(keys),'present':sum(k in fine for k in keys),'methods':{},'paired_comparisons':{}}
    def result(table,k,m):return table.get(k,{}).get('results',{}).get(m,{})
    for prefix,table in [('fine',fine),('coarse',coarse)]:
        for method in METHODS:
            rows=[result(table,k,method) for k in keys]
            valid=[r for r in rows if r.get('complete')]
            if any(type(r.get('complete')) is not bool or any(not np.isfinite(r[k]) or r[k]<0 for k in ('coordinate_rmse_m','shape_rmse_m')) for r in valid):raise ValueError('Invalid complete metric')
            out['methods'][prefix+'_'+method]={'expected':len(keys),'complete':len(valid),'coordinate_mean_mm':average([r['coordinate_rmse_m']*1000 for r in valid]),'shape_mean_mm':average([r['shape_rmse_m']*1000 for r in valid]),'conditional_on_completion':len(valid)!=len(keys),'failures':[{'object':k[0],'scenario':k[1],'reason':r.get('failure',r.get('reason','missing/incomplete'))} for k,r in zip(keys,rows) if not r.get('complete')]}
    pairs=[('coarse_fem',coarse,'fem')]+[('fine_'+m,fine,m) for m in METHODS if m!='fem']
    for label,other,method in pairs:
        gains=[];cases=[]
        for k in keys:
            a,b=result(fine,k,'fem'),result(other,k,method)
            if a.get('complete') and b.get('complete'):
                gain=(b['coordinate_rmse_m']-a['coordinate_rmse_m'])*1000;gains.append(gain);cases.append({'object':k[0],'scenario':k[1],'fine_fem_mm':a['coordinate_rmse_m']*1000,'comparator_mm':b['coordinate_rmse_m']*1000,'fine_gain_mm':gain})
        out['paired_comparisons'][label]={'expected':len(keys),'matched':len(gains),'gain_mm':average(gains),'wins':sum(g>1e-9 for g in gains),'losses':sum(g< -1e-9 for g in gains),'ties':sum(abs(g)<=1e-9 for g in gains),'cases':cases}
    return out

def parameter_diagnostics(protocol,selections):
    """Post-fit truth comparison only; never used for candidate selection."""
    output={'scope':'Post-fit point-estimate diagnostics, not posterior calibration. E errors apply only to FEM; native spring scale and rigid nominal scale are not Young modulus. Missing fits stay in coverage denominators.','splits':{}}
    for split in ('development','reserved'):
        objects=[o for o in protocol['objects'] if o['split']==split];rows=[];summaries={}
        for o in objects:
            row={'object':o['id'],'truth_young_pa':o['young'],'truth_friction':o['friction'],'fits':{}}
            for grid in ('coarse','fine'):
                for family in ('fem','native','rigid'):
                    r=(selections.get(grid,{}).get(o['id']) or {}).get(family,{})
                    fit={'success':bool(r.get('success',False)),'selected_scale':None,'selected_young_pa':None,'selected_friction':None,'calibration_rmse_m':None,'young_relative_abs_error':None,'friction_abs_error':None}
                    if fit['success']:
                        scale,mu,loss=r['scale'],r['friction'],r['calibration_rmse_m']
                        if not np.isfinite([scale,mu,loss]).all() or min(scale,mu,loss)<0:raise ValueError('Invalid selected parameters')
                        fit.update(selected_scale=scale,selected_young_pa=scale if family=='fem' else None,selected_friction=mu,calibration_rmse_m=loss,young_relative_abs_error=abs(scale/o['young']-1) if family=='fem' else None,friction_abs_error=abs(mu-o['friction']))
                    else:fit['failure']=r.get('reason','missing fit')
                    row['fits'][grid+'_'+family]=fit
            rows.append(row)
        for grid in ('coarse','fine'):
            for family in ('fem','native','rigid'):
                key=grid+'_'+family;valid=[r['fits'][key] for r in rows if r['fits'][key]['success']]
                summaries[key]={'expected':len(objects),'complete':len(valid),'conditional_on_available_fit':len(valid)!=len(objects),'mean_young_relative_abs_error':average([r['young_relative_abs_error'] for r in valid]) if family=='fem' else None,'mean_friction_abs_error':average([r['friction_abs_error'] for r in valid]),'mean_calibration_rmse_mm':average([r['calibration_rmse_m']*1000 for r in valid])}
        output['splits'][split]={'objects':rows,'summary':summaries}
    return output

def generate(report,original):
    p=read(report/'protocol.json');fine={};coarse={};hashes={};selections={'coarse':{},'fine':{}}
    for split in ('development','reserved'):
        for folder,lookup in [(report,fine),(original,coarse)]:
            path=folder/f'{split}-results.json'
            if not path.exists():continue
            hashes[str(path.relative_to(ROOT))]=sha(path)
            result=read(path);selections['fine' if folder==report else 'coarse'].update(result.get('selections',{}))
            for r in result['cases']:
                if r['split']!=split:raise ValueError('Split identity mismatch')
                key=(r['object'],r['scenario'])
                if key in lookup:raise ValueError('Duplicate case '+str(key))
                lookup[key]=r
    expected={(o['id'],s) for o in p['objects'] for s in p['forecast_scenarios']}
    if set(fine)-expected or set(coarse)-expected:raise ValueError('Undeclared cases')
    out={'scope':'Secondary parameter-grid sensitivity study; unchanged physics, observations and calibration loss. Not a new sealed primary study, public benchmark improvement, pure contact ablation, or full PhysTwin comparison. Native means the adapted author spring-force core. Stress remains development-unqualified. Positive paired gain favors fine-grid FEM.','source_hashes':hashes,'protocol_sha256':sha(report/'protocol.json'),'cohorts':{}}
    lines=[out['scope']]
    for split in ('development','reserved'):
        objects=[o for o in p['objects'] if o['split']==split];out['cohorts'][split]={}
        for name,scenes in [('primary',p['primary_forecast_scenarios']),('stress',p['stress_forecast_scenarios']),('full',p['forecast_scenarios'])]:
            keys=[(o['id'],s) for o in objects for s in scenes];v=summarize(keys,fine,coarse);v['scenarios']=scenes;out['cohorts'][split][name]=v
            lines.append(f'\n{split}/{name}: {v["present"]}/{v["expected"]} fixed cases; means conditional if failures')
            for m,r in v['methods'].items():lines.append(f'{m}: {r["complete"]}/{r["expected"]}, world {r["coordinate_mean_mm"]} mm, shape {r["shape_mean_mm"]} mm')
            for m,r in v['paired_comparisons'].items():lines.append(f'fine FEM versus {m}: {r["matched"]} matched; gain {r["gain_mm"]} mm; wins/losses/ties {r["wins"]}/{r["losses"]}/{r["ties"]}')
    out['development_primary_regressions']=out['cohorts']['development']['primary']['paired_comparisons']['coarse_fem']
    out['parameter_diagnostics']=parameter_diagnostics(p,selections)
    lines.extend(['','POST-FIT PARAMETER DIAGNOSTICS',out['parameter_diagnostics']['scope']])
    for split,d in out['parameter_diagnostics']['splits'].items():
        for row in d['objects']:
            lines.append(f'{split}/{row["object"]}: truth E={row["truth_young_pa"]} Pa, mu={row["truth_friction"]}')
            for method,f in row['fits'].items():lines.append(f'  {method}: success={f["success"]}, E={f["selected_young_pa"]}, scale={f["selected_scale"]}, mu={f["selected_friction"]}, calibration RMS m={f["calibration_rmse_m"]}')
        for method,r in d['summary'].items():lines.append(f'{split}/{method}: {r["complete"]}/{r["expected"]} fits; mean E relative absolute error={r["mean_young_relative_abs_error"]}; mean mu absolute error={r["mean_friction_abs_error"]}; conditional={r["conditional_on_available_fit"]}')
    audit=report/'independent-audit.json'
    out['library_coverage']=read(audit).get('library',{}) if audit.exists() else {}
    for family,r in out['library_coverage'].items():lines.append(f'Calibration {family}: {r["complete"]}/{r["expected"]} complete; {r["present"]} recorded; failures retained in audit candidate list')
    out['independent_audit']=dict(present=audit.exists(),passed=read(audit).get('passed',False) if audit.exists() else False,sha256=sha(audit) if audit.exists() else None)
    (report/'summary.json').write_text(json.dumps(out,indent=2)+'\n');(report/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=ROOT/'reports/coupled-friction-v13');parser.add_argument('--original',type=Path,default=ROOT/'reports/coupled-surface-v13');a=parser.parse_args();generate(a.report,a.original)
if __name__=='__main__':main()
