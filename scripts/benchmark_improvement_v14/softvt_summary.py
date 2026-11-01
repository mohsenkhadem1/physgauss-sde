"""Summarize the fixed development run, including regressions and phase errors."""
import json
from pathlib import Path
import numpy as np
from scripts.prefix_ridge_switch_v13.inputs import load_payload
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/benchmark-improvement-v14'
def main():
    result=json.loads((REPORT/'softvt-development-results.json').read_text());rows=result['cases'];out={'scope':'Development only; all20 fixed cases, no suffix-selected deployment','methods':{},'cases':[]}
    for method in ['proposal','model_selection_selected','prefix_calibrated_rigid']:
        scores=[r['proposal'] if method=='proposal' else r['baseline'][method] for r in rows]
        out['methods'][method]={k:float(np.mean([s['summary'][k] for s in scores])) for k in ['coordinate_rmse_mm','aligned_shape_rmse_mm']}
    for r in rows:
        name=r['case']['name'];data=load_payload(ROOT/'runs/prefix-ridge-switch-v13/inputs'/name/'full.npz');phase={}
        for method,score in [('proposal',r['proposal']),*r['baseline'].items()]:
            phase[method]={}
            for label,active in [('commanded_held',True),('commanded_released',False)]:
                frames=[f['coordinate_rmse_mm'] for f in score['frames'] if bool(data['attachment_active'][f['frame']].sum())==active]
                phase[method][label]=float(np.mean(frames)) if frames else None
        out['cases'].append(dict(name=name,trigger=r['metadata']['trigger'],speed_m_s=r['metadata'].get('speed_m_s'),world_gain_mm=r['baseline']['model_selection_selected']['summary']['coordinate_rmse_mm']-r['proposal']['summary']['coordinate_rmse_mm'],shape_gain_mm=r['baseline']['model_selection_selected']['summary']['aligned_shape_rmse_mm']-r['proposal']['summary']['aligned_shape_rmse_mm'],phase=phase))
    out['wins']=sum(r['world_gain_mm']>1e-9 for r in out['cases']);out['losses']=sum(r['world_gain_mm']< -1e-9 for r in out['cases']);out['ties']=len(rows)-out['wins']-out['losses']
    (REPORT/'softvt-development-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
