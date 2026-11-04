"""Summarize frozen-parameter inertia and numerical-step diagnostics."""
import json,numpy as np
from .development_run import ROOT,BASE,write

def main():
    development=json.loads((ROOT/'reports/real-transfer-v20/development-scores.json').read_text());rows=[]
    for case in ('single_lift_sloth','single_lift_zebra'):
        static=next(r for r in development['rows'] if r['case']==case and r['recipe']=='control')['metrics']['manual_track_mean_mm']
        result=dict(case=case,static_manual_mm=static);predictions=[]
        for n in (2,4):
            folder=BASE/'dynamic-diagnostic'/case/f'substeps{n}'
            meta=json.loads((folder/'complete.json').read_text());result[f'dynamic_substeps{n}_manual_mm']=meta['metrics']['manual_track_mean_mm']
            with np.load(folder/'prediction.npz') as f:predictions.append(f['predicted'])
        delta=np.linalg.norm(predictions[0]-predictions[1],axis=-1)*1000
        result.update(substep_mean_node_difference_mm=float(delta.mean()),substep_max_node_difference_mm=float(delta.max()))
        rows.append(result)
    write(ROOT/'reports/real-transfer-v20/dynamics-diagnostic.json',dict(rows=rows,
        scope='Fixed quasistatic parameters with implicit Euler inertia. Development prefixes only. Two substep counts are a sensitivity check, not a formal convergence proof.',
        conclusion='Adding inertia without refitting worsens validation error on both objects. Do not promote this variant; a dynamic inverse fit is a separate future experiment.'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
