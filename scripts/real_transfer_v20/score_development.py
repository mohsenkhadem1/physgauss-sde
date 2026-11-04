"""Validate only within the original training interval, never its test suffix."""
import json,numpy as np
from .development_campaign import CASES,RECIPES
from .development_run import ROOT,BASE,write,sha
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction,score

def main():
    rows=[]
    for case in CASES:
        for recipe,config in RECIPES.items():
            folder=BASE/'fits'/('development-'+recipe)/case/'neural'
            if not (folder/'complete.json').exists():
                rows.append(dict(case=case,recipe=recipe,status='unavailable'));continue
            complete=json.loads((folder/'complete.json').read_text());path=folder/'prediction.npz'
            if sha(path)!=complete['prediction_sha256']:raise ValueError('Modified forecast')
            data=phystwin(case,config['nodes']);start=complete['fit_end'];end=data['train_end']
            with np.load(path) as f:pred=expand_prediction(data,f['predicted'])
            if len(pred)!=end or not 0<start<end:raise ValueError('Wrong validation window')
            metric=score(data,pred,start,end)['summary']
            rows.append(dict(case=case,recipe=recipe,status='complete',interval=[start,end],metrics=metric))
    aggregate={}
    for recipe in RECIPES:
        group=[r for r in rows if r['recipe']==recipe and r['status']=='complete']
        valid=len(group)==len(CASES) and all(r['metrics']['valid_frames']==r['metrics']['total_frames'] for r in group)
        aggregate[recipe]=dict(eligible=valid,completed_cases=len(group),
            manual_track_mean_mm=float(np.mean([r['metrics']['manual_track_mean_mm'] for r in group])) if valid else None)
    candidates=[k for k,v in aggregate.items() if v['eligible']]
    complete=all(r['status']=='complete' for r in rows)
    selected=min(candidates,key=lambda k:aggregate[k]['manual_track_mean_mm']) if complete and candidates else None
    result=dict(rows=rows,aggregate=aggregate,complete=complete,selected_recipe=selected,
                scope='Development prefix validation only; no official suffix or cross-interaction performance claim.')
    write(ROOT/'reports/real-transfer-v20/development-scores.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
