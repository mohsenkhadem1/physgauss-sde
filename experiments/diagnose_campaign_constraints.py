"""TRAIN-only oracle-known constraint audit; never modifies a forward model.

Candidate controls only: old48 + expanded128 =176 unique TRAIN responses.
Exact fixture/attachment node IDs are used as diagnostic labels, NOT supplied
to existing teachers. No reserved/calibration/selection/gate/TEST response opens.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from campaign_control_data import load_control_snapshot
from train_gaussian_control_forward import collect_control_records
from train_gaussian_inverse import load_frozen_operator
from gaussian_operator import GaussianKernelFields
from isaac_dataset import boundary_shift


def constraint_metrics(positions, rest, times, spec):
    positions=np.asarray(positions,dtype=float);rest=np.asarray(rest,dtype=float)
    fixed=np.asarray(spec['fixture_nodes'],dtype=int)
    target=np.stack([rest[fixed]+boundary_shift(spec,t) for t in times])
    fixture_difference=positions[:,fixed]-target
    fixture_norm=np.linalg.norm(fixture_difference,axis=-1)
    patch_stats=[]
    for patch in spec['load_patches']:
        patch=np.asarray(patch,dtype=int);indices=np.triu_indices(len(patch),1)
        distances=np.linalg.norm(positions[:,patch,None]-positions[:,None,patch],axis=-1)
        base=np.linalg.norm(rest[patch,None]-rest[None,patch],axis=-1)
        errors=(distances-base)[:,indices[0],indices[1]]
        patch_stats.append({'pair_distance_rms_error_m':float(np.sqrt(np.mean(errors**2))),
                            'pair_distance_max_error_m':float(np.abs(errors).max())})
    return {'fixture_coordinate_rmse_m':float(np.sqrt(np.mean(fixture_difference**2))),
        'fixture_max_euclidean_error_m':float(fixture_norm.max()),
        'fixture_final_time_max_euclidean_error_m':float(fixture_norm[-1].max()),
        'fixture_per_time_max_euclidean_error_m':fixture_norm.max(-1).tolist(),
        'patches':patch_stats,
        'worst_patch_pair_distance_max_error_m':max(p['pair_distance_max_error_m'] for p in patch_stats),
        'mean_patch_pair_distance_rms_error_m':float(np.mean([p['pair_distance_rms_error_m'] for p in patch_stats]))}


def kernel_metrics(record,spec,fields):
    rest=torch.tensor(record['rest'][None]);physical=torch.tensor(record['physical'][None]);bc=torch.tensor([record['bc']])
    with torch.no_grad():values=fields(rest,physical,bc)
    support=values['support'][0].numpy().max(-1);mask=np.zeros(len(rest[0]),bool);mask[spec['fixture_nodes']]=True
    def overlap(kernel,mask):
        high=kernel>=.5
        return {'exact_nodes':int(mask.sum()),'kernel_on_exact_min':float(kernel[mask].min()),
            'kernel_on_exact_mean':float(kernel[mask].mean()),'kernel_on_exact_max':float(kernel[mask].max()),
            'nodes_above_half':int(high.sum()),'exact_nodes_below_half':int((mask&~high).sum()),
            'nonexact_nodes_above_half':int((~mask&high).sum()),
            'half_threshold_precision':float((mask&high).sum()/max(high.sum(),1)),
            'half_threshold_recall':float((mask&high).sum()/mask.sum())}
    result={'support':overlap(support,mask),'loads':[]}
    for i,patch in enumerate(spec['load_patches']):
        loadmask=np.zeros(len(mask),bool);loadmask[patch]=True
        result['loads'].append(overlap(values['load'][0,:,i].numpy(),loadmask))
    return result


def summarize(rows):
    def aggregate(selected):
        keys=['coordinate_rmse_m','fixture_coordinate_rmse_m','fixture_max_euclidean_error_m',
              'fixture_final_time_max_euclidean_error_m','worst_patch_pair_distance_max_error_m',
              'mean_patch_pair_distance_rms_error_m','total_prediction_squared_error_on_fixture_fraction',
              'oracle_fixture_only_replacement_coordinate_rmse_m']
        result={key:float(np.mean([r[key] for r in selected])) for key in keys if key in selected[0]}
        result['maximum_fixture_euclidean_error_m']=max(r['fixture_max_euclidean_error_m'] for r in selected)
        result['maximum_patch_pair_distance_error_m']=max(r['worst_patch_pair_distance_max_error_m'] for r in selected)
        return result
    return {'aggregate':aggregate(rows),'by_bc':{str(bc):aggregate([r for r in rows if r['bc']==bc]) for bc in range(4)},'episodes':rows}


def run(campaign,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    cfg=json.loads(Path(campaign).read_text())
    if cfg['new_train_control_ids']!=[f't{i:04d}' for i in range(16,48)]:raise ValueError('Fixed TRAIN role list changed')
    records=collect_control_records(Path(cfg['dataset']),Path(cfg['old_train_pool']),Path('configs/gaussian_probe_roles_v01.json'))
    records+=load_control_snapshot(cfg,cfg['new_train_control_ids'],'fit')
    if len(records)!=176 or any(r['split']!='train' for r in records):raise ValueError('Only exact176 TRAIN controls allowed')
    source_names=['diagnose_campaign_constraints.py','gaussian_operator.py','isaac_dataset.py','campaign_control_data.py','train_gaussian_control_forward.py']
    sources={name:Path(__file__).with_name(name).read_bytes() for name in source_names}
    (output/'provenance').mkdir()
    for name,content in sources.items():(output/'provenance'/name).write_bytes(content)
    specs={};fields=GaussianKernelFields();kernels={};ranks=[];actual=[]
    for r in records:
        eid=r['episode_id'].split('/')[0]
        if eid not in specs:
            specs[eid]=json.loads((Path(r['path']).parent/'truth/boundaries.json').read_text())
            kernels[eid]={'bc':int(r['bc']),**kernel_metrics(r,specs[eid],fields)}
        spec=specs[eid]
        actual.append({'episode_id':r['episode_id'],'bc':int(r['bc']),**constraint_metrics(r['q'],r['rest'],r['times'],spec)})
        singular=np.linalg.svd((r['q']-r['rest']).reshape(len(r['times']),-1).astype(float),compute_uv=False)
        energy=singular**2;total=energy.sum()
        ranks.append({'episode_id':r['episode_id'],'bc':int(r['bc']),
            'rank24_oracle_coordinate_rmse_floor_m':float(np.sqrt(energy[24:].sum()/np.prod(r['q'].shape))),
            'rank_for_99percent_energy':int(np.searchsorted(np.cumsum(energy),.99*total)+1),
            'rank_for_99p9percent_energy':int(np.searchsorted(np.cumsum(energy),.999*total)+1)})
    models={};started=time.monotonic()
    for name,path in [('original_passive',cfg['original_passive_forward']),('original_controlled',cfg['original_control_forward'])]:
        model=load_frozen_operator(path,'cpu');rows=[]
        with torch.no_grad():
            for r in records:
                predicted=model(torch.tensor(r['rest']),torch.tensor(r['times']),torch.tensor(r['physical'][None]),
                    torch.tensor([r['bc']]),torch.tensor([r['force']]))[0].numpy()
                spec=specs[r['episode_id'].split('/')[0]];metrics=constraint_metrics(predicted,r['rest'],r['times'],spec)
                error=(predicted.astype(float)-r['q'])**2;fixture=np.asarray(spec['fixture_nodes'])
                repaired=predicted.astype(float).copy()
                repaired[:,fixture]=np.stack([r['rest'][fixture]+boundary_shift(spec,t) for t in r['times']])
                rows.append({'episode_id':r['episode_id'],'bc':int(r['bc']),**metrics,
                    'coordinate_rmse_m':float(np.sqrt(error.mean())),
                    'total_prediction_squared_error_on_fixture_fraction':float(error[:,fixture].sum()/max(error.sum(),1e-30)),
                    'oracle_fixture_only_replacement_coordinate_rmse_m':float(np.sqrt(np.mean((repaired-r['q'])**2)))})
        models[name]={'checkpoint':path,'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),**summarize(rows)}
        print(json.dumps({'model':name,'aggregate':models[name]['aggregate']}),flush=True)
    result={'protocol':{'campaign_sha256':hashlib.sha256(Path(campaign).read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
        'unique_train_layouts':44,'unique_train_controls':176,'checkpoint_selection':False,
        'no_gate_calibration_selection_zero_diagonal_test_responses':True,
        'oracle_fixture_only_replacement':'TRAIN diagnostic lower-bound on editing only fixture positions, not a predictive method or a mechanical solve',
        'kernel_comparison':'Gaussian values are features, not claimed binary constraints; half threshold fixed only to characterize spatial extent',
        'temporal_rank':'rank24 oracle floor is capacity bound for stationary Green/single-response and raw rank24 operator; moving Green has two nonzero rank24 responses, zero-reference difference can rank48; no bound is asserted on nonlinear network parameter-to-basis mapping'},
        'models':models,'actual_train_constraints':summarize(actual),'kernel_layouts':kernels,'temporal_rank':ranks,
        'elapsed_model_evaluation_s':time.monotonic()-started}
    for n,b in sources.items():
        if Path(__file__).with_name(n).read_bytes()!=b:raise RuntimeError('Diagnostic source changed')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',default='configs/gaussian_campaign_v01.json');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.campaign,a.output)
