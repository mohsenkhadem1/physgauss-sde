"""Declared-cohort, object-level summaries from separately stored predictions."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.score import run as score,paired_interval
from scripts.position_ood.study import BASE,OUT,ROOT
from scripts.phystwin.sequential_reference import write

def summarize(split,method_names,indices=None):
    report=score(split);indices=list(range(report['declared_objects'])) if indices is None else list(indices);lookup={r['object']:r for r in report['rows']};groups={k:v for k,v in report['groups'].items() if k!='prefix'};rows=[]
    for method in method_names:
        missing=[i for i in indices if i not in lookup or method not in lookup[i]['methods']]
        entry=dict(method=method,declared_objects=len(indices),missing_objects=missing,complete=not missing,groups={})
        for group,actions in groups.items():
            objects=[i for i in indices if i not in missing];values=[float(np.mean(np.array(lookup[i]['methods'][method]['coordinate_rmse_mm'])[actions])) for i in objects]
            entry['groups'][group]=dict(objects=objects,per_object_rmse_mm=values,summary=paired_interval(values) if values else None)
        rows.append(entry)
    comparisons=[]
    if method_names:
        proposed=method_names[0]
        for baseline in method_names[1:]:
            for group,actions in groups.items():
                common=[i for i in indices if i in lookup and all(m in lookup[i]['methods'] for m in [proposed,baseline])];deltas=[];a=[];b=[]
                for i in common:
                    x=lookup[i]['methods'];a.append(float(np.mean(np.array(x[proposed]['coordinate_rmse_mm'])[actions])));b.append(float(np.mean(np.array(x[baseline]['coordinate_rmse_mm'])[actions])));deltas.append(b[-1]-a[-1])
                comparisons.append(dict(proposed=proposed,baseline=baseline,group=group,objects=common,complete=len(common)==len(indices),improvement_mm=paired_interval(deltas) if deltas else None,relative_mean_improvement_percent=100*(1-np.mean(a)/np.mean(b)) if b and np.mean(b)>0 else None))
    result=dict(split=split,declared_indices=indices,rows=rows,paired_comparisons=comparisons,aggregation='Mean action coordinate-RMSE within each object; equal-weight mean over objects; percentile paired bootstrap over objects. Positive paired difference favors the proposed model. Incomplete cohorts are labelled and do not substitute for declared complete results.')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--methods',nargs='+',default=['extrapolated-n3-p3','extrapolated_material_frame-n3-p3','extrapolated_nominal-n3-p3','neural-all','neural-id']);a=p.parse_args();r=summarize(a.split,a.methods);write(OUT/f'{a.split}-aggregate.json',r)
