"""Complete-case accounting, separated cohorts and numerical-reference status."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.coupled_contact_v13.campaign import REPORT,write
METHODS=['fem','native','rigid','position_ridge','position_rigid','persistence','oracle']
def summarize(rows):
    result={}
    for method in METHODS:
        valid=[r['results'][method] for r in rows if r.get('results',{}).get(method,{}).get('complete')]
        goodref=[r['results'][method] for r in rows if r.get('joint_reference_qualified') and r.get('results',{}).get(method,{}).get('complete')]
        failures=[dict(object=r['object'],scenario=r['scenario'],reason=r.get('results',{}).get(method,{}).get('failure',r.get('unavailable_reason','incomplete'))) for r in rows if not r.get('results',{}).get(method,{}).get('complete')]
        result[method]=dict(expected=len(rows),complete=len(valid),failures=failures,
            coordinate_mean_mm=float(np.mean([v['coordinate_rmse_m'] for v in valid])*1000) if valid else None,
            shape_mean_mm=float(np.mean([v['shape_rmse_m'] for v in valid])*1000) if valid else None,
            mean_conditional_on_complete_cases=len(valid)!=len(rows),qualified_calibration_and_forecast_count=len(goodref),
            qualified_reference_coordinate_mean_mm=float(np.mean([v['coordinate_rmse_m'] for v in goodref])*1000) if goodref else None)
    comparisons={}
    for method in METHODS[1:]:
        pairs=[(r['results']['fem']['coordinate_rmse_m'],r['results'][method]['coordinate_rmse_m']) for r in rows if r.get('results',{}).get('fem',{}).get('complete') and r.get('results',{}).get(method,{}).get('complete')]
        if pairs:
            p=np.array(pairs);gain=p[:,1]-p[:,0]
            comparisons[method]=dict(pairs=len(pairs),wins=int((gain>1e-12).sum()),losses=int((gain< -1e-12).sum()),ties=int((abs(gain)<=1e-12).sum()),mean_gain_mm=float(gain.mean()*1000))
    return dict(methods=result,fem_comparisons=comparisons)
def main():
    output={'scope':'Controlled synthetic cross-interaction transfer, same constitutive family and shared scene inputs. Native spring-core adaptation is not full PhysTwin. Empirical baselines lack scene collision geometry.','splits':{}}
    lines=['COUPLED CONTACT v13 RESULTS','Lower errors are better. Means are per sequence; repeated interactions share objects.']
    for split in ['development','validation','reserved']:
        path=REPORT/f'{split}-results.json'
        if not path.exists():continue
        d=json.loads(path.read_text());rows=d['cases'];summary=summarize(rows)
        summary.update(expected=d['expected_cases'],present=len(rows),reference_qualified=sum(bool(r.get('joint_reference_qualified')) for r in rows),
                       reference_failures=[dict(object=r['object'],scenario=r['scenario'],forecast_reasons=r.get('reference_qualification_reasons',[]),calibration_reasons=r.get('calibration_qualification_reasons',[])) for r in rows if not r.get('joint_reference_qualified')],
                       scenarios={s:summarize([r for r in rows if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows})},source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        output['splits'][split]=summary
        lines += ['',split.upper(),f"Declared {summary['expected']}; present {summary['present']}; reference qualified {summary['reference_qualified']}.", 'Method | complete | coordinate mm | rigid-aligned shape mm']
        for method,s in summary['methods'].items():lines.append(f"{method} | {s['complete']}/{s['expected']} | {s['coordinate_mean_mm']} | {s['shape_mean_mm']}")
        lines.append('Means with incomplete coverage are conditional. Unqualified references remain in all-case accounting; jointly qualified calibration/forecast means are separately recorded.')
    write(REPORT/'summary.json',output);(REPORT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
