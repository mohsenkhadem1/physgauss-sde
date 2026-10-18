"""Observed-data-only feasibility gate; never a convergence/calibration claim."""
import argparse
import json
from pathlib import Path
from pacnerf_comparison import write


def check(root,output):
    records=[]
    for index in [0,1,2,3,8,9,10,11]:
        paths=list((Path(root)/str(index)).glob('**/pilot_summary.json'))
        if len(paths)!=1:raise ValueError(f'Missing/ambiguous pilot {index}')
        record=json.loads(paths[0].read_text())
        if record['settings']['case_index']!=index or not record['pilot'] or record['future_geometry_used']:
            raise ValueError('Pilot identity or no-leakage assertion failed')
        if record['draws_per_chain']!=100 or not record['replay']['all_pass']:
            raise ValueError('Incomplete pilot or failed target replay')
        if record['sampler']!='elliptical' or record['valid_for_ranking']:
            raise ValueError('Unexpected pilot protocol')
        if any(record['retained_invalid']):raise ValueError('Invalid retained pilot states')
        fit=json.loads((paths[0].parent/'fit.json').read_text())['stationarity']
        if not fit['stationary'] or fit['gauss_newton_decrement']>=.02 or fit['history'][-1]['gradient_max_abs']>=.02:
            raise ValueError('Reference fit is not stationary')
        records.append(record)
    result={'all_observed_only_feasibility_checks_pass':True,'pilots':records,
        'qualification':'100-draw development runs test numerical feasibility and exact target replay; no convergence, reconstruction or calibration claim; full runs start afresh'}
    write(output,result);print(json.dumps({'pilot_count':len(records),'all_pass':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();check(a.root,a.output)
