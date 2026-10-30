"""Prespecified fixed-parameter FEM supplement; does not modify frozen forecasts."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.pipeline import check_freeze,simulate_job,metrics,shared_node_indices
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write
from scripts.coupled_contact_v13.scenarios import build

def main():
    check_freeze();p=json.loads((REPORT/'protocol.json').read_text());supp=json.loads((REPORT/'supplementary-nominal-protocol.json').read_text())
    import hashlib
    for path,sha in supp['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Supplement source changed: '+path)
    rows=[]
    for scenario in p['forecast_scenarios']:
        job=dict(path=f'supplementary-nominal/{scenario}',family='fem',scenario=scenario,scale=supp['young_pa'],friction=supp['friction'],reference=False)
        simulate_job(job)
    for split in ['development','validation','reserved']:
        main_result=json.loads((REPORT/f'{split}-results.json').read_text());main_rows={(r['object'],r['scenario']):r for r in main_result['cases']}
        for obj in [o for o in p['objects'] if o['split']==split]:
            for scenario in p['forecast_scenarios']:
                main_row=main_rows[(obj['id'],scenario)];reference=BASE/'reference'/obj['id']/scenario
                status=json.loads((reference/'complete.json').read_text());path=BASE/'supplementary-nominal'/scenario;pred_status=json.loads((path/'complete.json').read_text())
                row=dict(object=obj['id'],split=split,scenario=scenario,reference_generation_success=status['success'],joint_reference_qualified=main_row['joint_reference_qualified'],prediction_path=str(path.relative_to(ROOT)),results={})
                if not status['success']:
                    row['unavailable_reason']='Reference generation failed';rows.append(row);continue
                coarse=build(scenario,cells=tuple(p['coarse_cells']))
                with np.load(reference/'trajectory.npz') as d:
                    target=d['frames'][:,shared_node_indices(d['rest'],coarse.rest)];body=d['centers']
                pred=None
                if pred_status['success']:
                    with np.load(path/'trajectory.npz') as d:pred=d['frames'];predbody=d['centers']
                result=metrics(pred,target)
                if pred_status['success']:
                    result['diagnostics']=pred_status['diagnostics']
                    if scenario=='moving_sphere':result['dynamic_body_position_rmse_m']=float(np.sqrt(np.mean(np.sum((predbody[1:,-1]-body[1:,-1])**2,axis=1))))
                else:result['failure']=pred_status.get('error','Nominal forward model failed')
                row['results']['nominal_fem']=result;fitted=main_row['results'].get('fem',{})
                if result['complete'] and fitted.get('complete'):
                    row['inference_gain_mm']=(result['coordinate_rmse_m']-fitted['coordinate_rmse_m'])*1000
                rows.append(row)
    summary={}
    for split in ['development','validation','reserved']:
        cohort=[r for r in rows if r['split']==split];valid=[r['results']['nominal_fem'] for r in cohort if r['results'].get('nominal_fem',{}).get('complete')];gain=np.array([r['inference_gain_mm'] for r in cohort if 'inference_gain_mm' in r])
        summary[split]=dict(expected=len(cohort),complete=len(valid),mean_coordinate_mm=float(np.mean([r['coordinate_rmse_m'] for r in valid])*1000) if valid else None,
            paired_cases=len(gain),inference_mean_gain_mm=float(gain.mean()) if len(gain) else None,inference_wins=int((gain>1e-9).sum()),inference_losses=int((gain< -1e-9).sum()),ties=int((abs(gain)<=1e-9).sum()))
    write(REPORT/'nominal-baseline-results.json',dict(scope=supp['purpose'],expected_cases=56,cases=rows,splits=summary))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
