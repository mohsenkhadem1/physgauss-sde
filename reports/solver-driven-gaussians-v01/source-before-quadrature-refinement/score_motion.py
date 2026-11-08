"""Read sealed predictions and report complete-case geometry, including failures."""
from pathlib import Path
import json,hashlib
import numpy as np
BASE=Path('runs/solver-driven-gaussians-v01');REPORT=Path('reports/solver-driven-gaussians-v01')

def metrics(pred,truth,rest):
    def align(x,y):
        a=x-x.mean(0);b=y-y.mean(0);U,_,Vt=np.linalg.svd(a.T@b);R=U@Vt
        if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
        return a@R-b
    return dict(position_rms_mm=float(np.sqrt(np.mean(np.sum((pred-truth)**2,-1)))*1000),center_rms_mm=float(np.sqrt(np.mean(np.sum((pred.mean(1)-truth.mean(1))**2,-1)))*1000),rigid_aligned_error_mm=float(np.sqrt(np.mean([np.mean(np.sum(align(x,y)**2,-1)) for x,y in zip(pred,truth)]))*1000),reference_deformation_rms_mm=float(np.sqrt(np.mean([np.mean(np.sum(align(rest,y)**2,-1)) for y in truth]))*1000))

def main():
    rows=[]
    for method in ['oracle','analytic','rgb','rgb-refined']:
        folder=BASE/(method+'-predictions')
        if not folder.exists():continue
        for case in ['calibration','side_poke','early_release','force_pulse','higher_drop']:
            p=folder/(case+'.npz');seal=folder/(case+'.json')
            if not p.exists() or not seal.exists():rows.append(dict(method=method,case=case,completed=False));continue
            d=json.loads(seal.read_text());assert d['completed'] and not d['target_states_read'];assert hashlib.sha256(p.read_bytes()).hexdigest()==d['prediction_sha256']
            pred=np.load(p);truth=np.load(BASE/'reference'/(case+'.npz'));assert np.allclose(pred['times'],truth['times']);keep=truth['times']>.14
            rows.append(dict(method=method,case=case,completed=True,**metrics(pred['x'][keep],truth['x'][keep],truth['rest']),minimum_J=d['minimum_J'],penetration_mm=1000*d['max_penetration_m']))
    sensitivity=[]
    for case in ['calibration','higher_drop']:
        coarse=np.load(BASE/'reference'/(case+'.npz'));fine=np.load(BASE/'reference-fine'/(case+'.npz'));sample=fine['x'].reshape(-1,17,9,9,3)[:,::2,::2,::2].reshape(-1,225,3)
        sensitivity.append(dict(case=case,**metrics(coarse['x'],sample,coarse['rest'])))
    out=dict(rows=rows,spatial_reference_sensitivity=sensitivity,scope='Changed-control forecasts after 0.14 s. Oracle means supplied calibration positions, not known physical parameters. Numerical references; one object/material, not a dataset-wide claim.')
    (REPORT/'motion-results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
