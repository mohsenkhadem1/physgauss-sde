"""Summarize completed pilots and diagnose deformation independently of rendering."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
try:
    from .run_physics import metrics
except ImportError:  # Direct command-line execution.
    from run_physics import metrics


def rigid_residual(reference, target):
    x=reference-reference.mean(axis=0)
    y=target-target.mean(axis=0)
    u,_,vt=np.linalg.svd(x.T@y)
    correction=np.eye(3); correction[-1,-1]=np.linalg.det(u@vt)
    fitted=x@(u@correction@vt)+target.mean(axis=0)
    return float(np.sum((target-fitted)**2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/third_party/PhysTwin'))
    p.add_argument('--runs',type=Path,default=Path('runs/phystwin-pilot-v01'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result={}
    for directory in [a.runs/name for name in
                      ['reference','double-reference','retrained','fresh-cma','fresh-two-stage-20']]:
        if not (directory/'summary.json').exists(): continue
        summary=json.loads((directory/'summary.json').read_text())
        data_dir=a.root/'data/different_types'/summary['case']
        with (data_dir/'gt_track_3d.pkl').open('rb') as f: track=pickle.load(f)
        with (data_dir/'final_data.pkl').open('rb') as f: data=pickle.load(f)
        with (directory/'inference.pkl').open('rb') as f: vertices=pickle.load(f)
        cutoff=summary['split']['train'][1]
        persistence=vertices.copy();persistence[cutoff:]=vertices[cutoff-1]
        # Constant velocity is estimated only from the final two prefix states.
        linear=vertices.copy();velocity=vertices[cutoff-1]-vertices[cutoff-2]
        for t in range(cutoff,len(vertices)):
            linear[t]=vertices[cutoff-1]+(t-cutoff+1)*velocity
        raw_sq=elastic_sq=0.;count=0
        for t in range(1,len(track)):
            valid=np.isfinite(track[0]).all(1)&np.isfinite(track[t]).all(1)
            if valid.sum()<3: continue
            x,y=track[0,valid],track[t,valid]
            raw_sq+=float(np.sum((y-x)**2));elastic_sq+=rigid_residual(x,y);count+=len(x)
        result[directory.name]={
            'case':summary['case'],
            'physical_model':summary['result']['test'],
            'last_prefix_model_state_persistence':metrics(persistence,data,summary['split'],track)['test'],
            'last_prefix_model_velocity_extrapolation':metrics(linear,data,summary['split'],track)['test'],
            'manual_motion_rms_m':float(np.sqrt(raw_sq/count)),
            'manual_nonrigid_rms_m':float(np.sqrt(elastic_sq/count)),
            'nonrigid_to_total_rms_ratio':float(np.sqrt(elastic_sq/raw_sq)),
            'note':'Temporal baselines use the same fitted prefix states and no future object observations. Nonrigidity uses sparse manual tracks across the full clip and is a diagnostic only; not a dense strain or elasticity estimate.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
