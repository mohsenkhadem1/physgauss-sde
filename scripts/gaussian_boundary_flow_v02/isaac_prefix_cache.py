"""Role-checked five-frame cache. No test cases or trajectory files are read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from train_gaussian_inverse import collect_cases,apply_catalog,load_frozen_operator
from evaluate_gaussian_inverse import load_public_observations,predict_particles
from gaussian_observed_features import extract_features
from gaussian_operator import physical_vector,category_labels,canonicalize_physical


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def cache_row(item):
    if item['recipe']['split'] not in ('train','validation'):
        raise ValueError('Only TRAIN/VALIDATION cache access permitted')
    episode=Path(item['path'])
    inputs=load_public_observations(episode,'rgbd',size=(192,160),max_time_s=1.0)
    times=np.asarray(inputs['cameras']['times_s'])
    if times.shape!=(5,) or not np.allclose(times,np.arange(5)*.25,rtol=0,atol=1e-7):
        raise ValueError('Expected exactly five declared prefix times')
    measured=extract_features(inputs)['features'].astype(np.float32)
    if measured.shape!=(5,119) or not np.isfinite(measured).all():raise ValueError('Invalid prefix features')
    spec_path=episode/'truth/boundaries.json'
    spec=json.loads(spec_path.read_text())
    physical=physical_vector(item['recipe'],spec,inputs['reference']['rest_points_m'])
    bc,force=category_labels(item['recipe'])
    paths=[episode/'reference.npz',episode/'observations/cameras.npz',spec_path]
    for frame in inputs['cameras']['frame_ids']:
        paths.extend(episode/'observations'/folder/f'{frame:05d}.{suffix}' for folder,suffix in [('rgb','png'),('masks','png'),('depth','npz')])
    arrays=dict(features=measured,physical=np.asarray(physical,dtype=np.float32),boundary=np.int64(bc),force=np.int64(force),
                rest=np.asarray(inputs['reference']['rest_points_m'],dtype=np.float32),times=times.astype(np.float32),
                frame_ids=np.asarray(inputs['cameras']['frame_ids']))
    return arrays,{str(p.resolve()):sha(p) for p in paths}


def build(catalog,output):
    catalog=Path(catalog);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads(catalog.read_text());cases=apply_catalog(collect_cases(data['roots']),catalog)
    # Admission before any feature/label loading.
    if len(cases)!=196 or sum(c['recipe']['split']=='train' for c in cases)!=152:
        raise ValueError('Frozen 152 TRAIN/44 VAL catalog required')
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/evaluate_gaussian_inverse.py',ROOT/'experiments/train_gaussian_inverse.py',ROOT/'experiments/gaussian_observed_features.py',ROOT/'experiments/gaussian_operator.py',ROOT/'experiments/gaussian_observation.py']}
    contract=dict(catalog_sha256=sha(catalog),sources=sources,prefix_times_s=[0,.25,.5,.75,1.],feature_shape=[5,119],no_test_access=True,no_trajectory_access=True)
    contract_path=output/'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:raise ValueError('Cache contract changed')
    else:write_json(contract_path,contract)
    rows=[]
    for index,item in enumerate(cases):
        name=f'{index:03d}.npz';path=output/name;meta=output/f'{index:03d}.json'
        if path.exists() or meta.exists():
            if not(path.exists() and meta.exists()):raise ValueError('Incomplete shard; inspect before replacing')
            row=json.loads(meta.read_text())
            if row['cache_sha256']!=sha(path) or row['key']!=item['key']:raise ValueError('Shard mismatch')
            if any(sha(p)!=v for p,v in row['input_sha256'].items()):raise ValueError('Input changed')
        else:
            arrays,hashes=cache_row(item);np.savez_compressed(path,**arrays)
            row=dict(key=item['key'],role=item['recipe']['split'],group=item['recipe']['group_id'],dataset=item['dataset'],
                     recipe_sha256=item['recipe']['recipe_sha256'],cache=name,cache_sha256=sha(path),input_sha256=hashes)
            write_json(meta,row)
        rows.append(row)
        write_json(output/'progress.json',dict(completed=index+1,total=len(cases)))
    manifest=dict(contract_sha256=sha(contract_path),cases=rows)
    write_json(output/'manifest.json',manifest)
    return manifest


def smoke(cache,teacher,output):
    import torch
    torch.set_num_threads(2)
    cache=Path(cache);m=json.loads((cache/'manifest.json').read_text());row=next(r for r in m['cases'] if r['role']=='train')
    if row['role']!='train':raise ValueError('Smoke restricted to TRAIN')
    path=cache/row['cache']
    if sha(path)!=row['cache_sha256']:raise ValueError('Cache changed')
    with np.load(path) as f:arrays={k:f[k] for k in f.files}
    forward=load_frozen_operator(teacher,'cpu');physical=torch.tensor(arrays['physical'][None]);bc=torch.tensor([int(arrays['boundary'])]);force=torch.tensor([int(arrays['force'])])
    particles=dict(physical=physical,effective_physical=canonicalize_physical(physical,bc,force),boundary_type=bc,force_type=force)
    prediction=predict_particles(dict(device='cpu',forward=forward),particles,dict(rest_points_m=arrays['rest']),np.array([0,1,2,2.5,3],np.float32))
    if not torch.isfinite(prediction).all() or prediction.shape!=(1,5,len(arrays['rest']),3):raise ValueError('Teacher contract failed')
    report=dict(case=row['key'],role='train',teacher_sha256=sha(teacher),cache_sha256=sha(path),shape=list(prediction.shape),finite=True,
                max_displacement_m=float((prediction-torch.tensor(arrays['rest'])).abs().max()),
                interpretation='API smoke using TRAIN physical labels; no trajectory comparison and no accuracy claim')
    write_json(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',default='runs/gaussian-final-passive-catalog.json');p.add_argument('--output',default='runs/gaussian-boundary-isaac-prefix-v01');p.add_argument('--smoke',action='store_true');p.add_argument('--teacher',default='runs/gaussian-inverse-final-forward.pt');a=p.parse_args()
    if a.smoke:print(json.dumps(smoke(a.output,a.teacher,Path(a.output)/'teacher-smoke.json')))
    else:build(a.catalog,a.output)
