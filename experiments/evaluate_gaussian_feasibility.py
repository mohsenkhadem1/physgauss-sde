"""Paired raw versus explicitly feasibility-conditioned validation predictions."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from evaluate_gaussian_inverse import load_model,encode_episode
from train_gaussian_inverse import collect_cases,apply_catalog
from gaussian_operator import deformation_diagnostics
from gaussian_feasibility import feasible_posterior_samples


def score(q,target,reference):
    mean=q.mean(0);lower,upper=np.quantile(q,[.05,.95],axis=0)
    rest=reference['rest_points_m'];tets=reference['tetrahedra'];gap=np.arange(61)/20
    mask=(gap>1)&(gap<2)
    result={'coordinate_rmse_m':float(np.sqrt(np.mean((mean-target)**2))),
            'gap_coordinate_rmse_m':float(np.sqrt(np.mean((mean[mask]-target[mask])**2))),
            'coverage90':float(((target[1:]>=lower[1:])&(target[1:]<=upper[1:])).mean()),
            'mean_width90_m':float((upper[1:]-lower[1:]).mean())}
    check=deformation_diagnostics(torch.tensor(mean),torch.tensor(rest,dtype=torch.float32),torch.tensor(tets,dtype=torch.long))
    result['mean_trajectory_diagnostics']=check
    return result,{'mean':mean,'lower90':lower,'upper90':upper}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--catalog',required=True);parser.add_argument('--inverse-checkpoint',required=True)
    parser.add_argument('--forward-checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,default=64);parser.add_argument('--budget',type=int,default=256)
    parser.add_argument('--device',default='cpu');args=parser.parse_args();torch.set_num_threads(2)
    bundle=load_model(args.inverse_checkpoint,args.forward_checkpoint,args.device)
    records=[r for r in apply_catalog(collect_cases(args.roots),args.catalog) if r['recipe']['split']=='validation']
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);rows=[]
    provenance=output/'provenance';provenance.mkdir();hashes={}
    for name in ('evaluate_gaussian_feasibility.py','gaussian_feasibility.py','evaluate_gaussian_inverse.py','train_gaussian_inverse.py','gaussian_operator.py','gaussian_observed_features.py'):
        data=Path(__file__).with_name(name).read_bytes();(provenance/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    for index,record in enumerate(records):
        encoded=encode_episode(bundle,record['path']);reference=encoded['inputs']['reference']
        times=np.arange(61)/20
        draws=feasible_posterior_samples(bundle,encoded,reference,times,args.samples,args.budget,20261102+index)
        # Hidden motion is read only after all feasibility decisions are frozen.
        with np.load(record['path']/'truth/trajectory.npz') as raw:target=raw['positions_m']
        raw_score,raw_arrays=score(draws['raw_predictions'],target,reference)
        retained_score=None;retained_arrays={}
        if draws['predictions'] is not None and len(draws['predictions'])>=2:
            retained_score,retained_arrays=score(draws['predictions'],target,reference)
        row={'key':record['key'],'raw':raw_score,'feasible_conditioned':retained_score,'sampler':draws['stats']}
        np.savez_compressed(output/(record['recipe']['episode_id']+'.npz'),
                            **{'raw_'+k:v for k,v in raw_arrays.items()},**{'conditioned_'+k:v for k,v in retained_arrays.items()},
                            candidate_valid=draws['candidate_valid'],candidate_minimum_volume_ratio=draws['candidate_minimum_volume_ratio'])
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='sampler'}),flush=True)
    aggregate={}
    for mode in ['raw','feasible_conditioned']:
        eligible=[r[mode] for r in rows if r[mode] is not None]
        aggregate[mode]={key:float(np.mean([r[key] for r in eligible])) for key in ['coordinate_rmse_m','gap_coordinate_rmse_m','coverage90','mean_width90_m']}
    aggregate['estimated_retained_mass_mean']=float(np.mean([r['sampler']['estimated_raw_retained_probability'] for r in rows]))
    aggregate['insufficient_support_episodes']=sum(r['sampler']['insufficient_support'] for r in rows)
    result={'episodes':rows,'aggregate':aggregate,'inverse_checkpoint':args.inverse_checkpoint,'forward_checkpoint':args.forward_checkpoint,
            'source_sha256':hashes,'checkpoint_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('inverse',args.inverse_checkpoint),('forward',args.forward_checkpoint)]},
            'catalog':args.catalog,'qualification':'Paired first64 raw versus first64 grid-feasible candidates from256; no test cases or hidden-state particle selection'}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
