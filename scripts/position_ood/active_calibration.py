"""Separate pre/post acquisition calibration and spatial-transfer scoring."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT
from scripts.position_ood.calibration import standardized_score,conformal_scale,coverage
from scripts.position_ood.score import paired_interval
from scripts.position_ood.seal import verify,digest
from scripts.phystwin.sequential_reference import write
POLICIES=['fixed','random','adaptive'];TARGET=7

def arrays(split,i,policy=None):
    directory=BASE/f'inverse/{split}-{i}';tag='p3';choice=json.loads((BASE/f'active-jobs/{split}-{i}-choices.json').read_text())
    if policy is not None:tag=f'p4-0-1-2-{choice["actions"][policy]}'
    path=directory/f'extrapolated-n3-{tag}-uncertainty.npz';reference=BASE/f'fields/{split}-{i}.npz'
    with np.load(path) as p,np.load(reference) as r:values=(p['mean'][TARGET].copy(),p['std'][TARGET].copy(),r['field'][TARGET].copy())
    return values,[path,reference],choice

def calibrate():
    verify();scores={k:[] for k in ['before',*POLICIES]};sources={}
    for i in range(12):
        for key in scores:
            (mean,std,truth),paths,_=arrays('transfer_calibration',i,None if key=='before' else key);scores[key].append(standardized_score(mean,std,truth));sources.update({str(p.relative_to(BASE)):digest(p) for p in paths})
        p=BASE/f'active-jobs/transfer_calibration-{i}-choices.json';sources[str(p.relative_to(BASE))]=digest(p)
    record=dict(objects=12,target=TARGET,scales={k:conformal_scale(v) for k,v in scores.items()},scores=scores,sources=sources,scope='Separate acquisition-conditioned scales on independent objects. Calibration target responses enter interval scaling only, never test fits or probe selection. Both pre/post intervals use the same 90%-coordinate / 90%-object rule.')
    path=BASE/'calibration/transfer.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise RuntimeError('Frozen transfer calibration inputs changed')
    else:write(path,record)
    return record

def evaluate(split='transfer_test'):
    if split=='transfer_test':verify()
    cal=json.loads((BASE/'calibration/transfer.json').read_text())
    for name,h in cal['sources'].items():
        if digest(BASE/name)!=h:raise RuntimeError('Frozen transfer calibration source changed')
    count=9 if split=='transfer_test' else 3;rows=[];missing=[]
    for i in range(count):
        try:
            before,_,choice=arrays(split,i);row=dict(object=i,actions=choice['actions'],before=coverage(*before,cal['scales']['before']),after={})
            for policy in POLICIES:
                values,_,_=arrays(split,i,policy);row['after'][policy]=coverage(*values,cal['scales'][policy])
            rows.append(row)
        except FileNotFoundError as error:missing.append(dict(object=i,error=str(error)))
    ablation_rows=[];ablation_missing=[]
    for i in range(count):
        try:
            choice=json.loads((BASE/f'active-jobs/{split}-{i}-choices.json').read_text());action=choice['actions']['adaptive'];directory=BASE/f'inverse/{split}-{i}'
            with np.load(BASE/f'fields/{split}-{i}.npz') as r:truth=r['field'][TARGET].copy()
            result=dict(object=i,action=action)
            for frame,method in [('world','extrapolated'),('material','extrapolated_material_frame')]:
                result[frame]={}
                for phase,tag in [('before','p3'),('after',f'p4-0-1-2-{action}')]:
                    with np.load(directory/f'{method}-n3-{tag}.npz') as p:result[frame][phase]=float(1000*np.sqrt(np.mean((p['predicted'][TARGET]-truth)**2)))
            ablation_rows.append(result)
        except FileNotFoundError as error:ablation_missing.append(dict(object=i,error=str(error)))
    ablation=dict(rows=ablation_rows,missing=ablation_missing,complete=not ablation_missing,world_advantage_mm={phase:paired_interval([r['material'][phase]-r['world'][phase] for r in ablation_rows]) for phase in ['before','after']} if ablation_rows else {},scope='Same acquired response, same inverse solver, different coordinate frame for environmental loading.')
    summary={}
    for policy in POLICIES:
        errors=[r['after'][policy]['rmse_mm'] for r in rows];improvements=[r['before']['rmse_mm']-r['after'][policy]['rmse_mm'] for r in rows];summary[policy]=dict(after_rmse=paired_interval(errors) if errors else None,improvement=paired_interval(improvements) if improvements else None,coordinate_coverage=float(np.mean([r['after'][policy]['coordinate_coverage'] for r in rows])) if rows else None,mean_half_width_mm=float(np.mean([r['after'][policy]['mean_half_width_mm'] for r in rows])) if rows else None)
    comparisons={policy:paired_interval([r['after'][policy]['rmse_mm']-r['after']['adaptive']['rmse_mm'] for r in rows]) for policy in ['fixed','random']} if rows else {}
    result=dict(split=split,declared_objects=count,complete=not missing,missing=missing,rows=rows,summary=summary,adaptive_advantage_mm=comparisons,calibration_scales=cal['scales'],matched_frame_ablation=ablation,scope='Object-level comparisons on a separately held spatial target. Positive paired advantage favors adaptive acquisition. No superiority claim follows from ties or intervals spanning zero.')
    write(OUT/f'{split}-active-results.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['calibrate','evaluate'],default='calibrate');p.add_argument('--split',default='transfer_test');a=p.parse_args();print(calibrate() if a.task=='calibrate' else evaluate(a.split))
