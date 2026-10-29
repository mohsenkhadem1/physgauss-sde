"""Object-level split calibration for marginal coordinate prediction intervals.

A calibration object's score is its 90th percentile standardized coordinate
error across declared held actions. The finite-sample order statistic targets
90% of exchangeable objects having at least 90% of coordinates covered. OOD
coverage is measured, not guaranteed. No test residual adjusts the scale.
"""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,protocol
from scripts.phystwin.sequential_reference import write

FLOOR_M=2e-5

def standardized_score(mean,std,truth,coordinate_fraction=.9):
    scale=np.sqrt(np.asarray(std)**2+FLOOR_M**2)
    return float(np.quantile(np.abs(np.asarray(mean)-truth)/scale,coordinate_fraction,method='higher'))

def conformal_scale(scores,object_fraction=.9):
    x=np.asarray(scores,float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty calibration scores required')
    k=math.ceil((len(x)+1)*object_fraction)
    return float(np.sort(x)[k-1]) if k<=len(x) else float('inf')

def coverage(mean,std,truth,q):
    width=q*np.sqrt(std**2+FLOOR_M**2);covered=np.abs(mean-truth)<=width
    return dict(coordinate_coverage=float(covered.mean()),at_least_90_percent_coordinates=bool(covered.mean()>=.9),mean_half_width_mm=float(1000*width.mean()),rmse_mm=float(1000*np.sqrt(np.mean((mean-truth)**2))))

def calibrate(method='extrapolated',mesh=3):
    from scripts.position_ood.seal import verify,digest
    verify();actions=np.arange(3,10);scores=[];sources={}
    for i in range(len(protocol()['calibration'])):
        p=BASE/f'inverse/calibration-{i}/{method}-n{mesh}-p3-uncertainty.npz';r=BASE/f'fields/calibration-{i}.npz'
        with np.load(p) as x,np.load(r) as ref:scores.append(standardized_score(x['mean'][actions],x['std'][actions],ref['field'][actions]))
        sources[str(p.relative_to(BASE))]=digest(p);sources[str(r.relative_to(BASE))]=digest(r)
    q=conformal_scale(scores);record=dict(method=method,mesh=mesh,actions=actions.tolist(),objects=len(scores),scores=scores,scale=q,coordinate_fraction=.9,object_fraction=.9,floor_m=FLOOR_M,sources=sources,scope='Split calibration on independent objects; exchangeability guarantee does not extend to OOD cohorts.')
    dest=BASE/f'calibration/{method}-n{mesh}.json'
    if dest.exists():
        if json.loads(dest.read_text())!=record:raise RuntimeError('Calibration already frozen with different inputs')
    else:write(dest,record)
    return record

def evaluate(split,method='extrapolated',mesh=3):
    from scripts.position_ood.seal import verify,digest
    verify();cal=json.loads((BASE/f'calibration/{method}-n{mesh}.json').read_text())
    for name,h in cal['sources'].items():
        if digest(BASE/name)!=h:raise RuntimeError('Frozen calibration source changed')
    rows=[]
    for i in range(len(protocol()[split])):
        p=BASE/f'inverse/{split}-{i}/{method}-n{mesh}-p3-uncertainty.npz'
        with np.load(p) as x,np.load(BASE/f'fields/{split}-{i}.npz') as ref:
            rows.append(dict(object=i,**coverage(x['mean'][3:],x['std'][3:],ref['field'][3:],cal['scale'])))
    report=dict(split=split,method=method,objects=len(rows),calibration_scale=cal['scale'],rows=rows,mean_coordinate_coverage=float(np.mean([r['coordinate_coverage'] for r in rows])),fraction_objects_at_least_90_percent_covered=float(np.mean([r['at_least_90_percent_coordinates'] for r in rows])))
    write(OUT/f'{split}-{method}-coverage.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='calibration');a=p.parse_args();print(calibrate() if a.split=='calibration' else evaluate(a.split))
