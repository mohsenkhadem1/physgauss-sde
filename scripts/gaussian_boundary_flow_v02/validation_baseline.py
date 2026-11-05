"""Seal standalone prefix baselines before geometry/backend qualification."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.object_score import prefix_baselines,sha
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.softvt_validation_inputs import admission
from scripts.external_benchmark.data import fps


def prepare(package,output,protocol_path,protocol_sha256):
    package,output=Path(package),Path(output)
    if output.exists():raise FileExistsError('Immutable baseline run exists')
    data=load_package(package)
    if data.get('data_role')!='prospective_validation'or data.get('validation_protocol_sha256')!=protocol_sha256:raise ValueError('Baseline input admission mismatch')
    import re
    match=re.fullmatch(r'(object-soft|spatial-soft)-task([012])-demo([0-9]+)',data['name'])
    if match is None:raise ValueError('Unknown case')
    admission(match[1],int(match[2]),int(match[3]),protocol_path=protocol_path,protocol_sha256=protocol_sha256)
    frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    baselines=prefix_baselines(data,frames,ids)
    output.mkdir(parents=True)
    plan={'case':data['name'],'dataset':data['dataset'],'input_package':str(package.resolve()),'input_package_manifest_sha256':sha(package/'manifest.json'),
          'input_digest':predictor_digest(data),'args':{'nodes':len(data['nodes'])},'prefix_frames':frames.tolist(),'point_ids':ids.tolist(),
          'prediction_methods':['initial_geometry'],'data_role':'prospective_validation','validation_protocol_sha256':protocol_sha256,
          'scope':'Standalone observed-prefix persistence/velocity plus initial geometry; independent of mechanics success. Exact fullpoint baselines stored and scorer verifies them.'}
    write=lambda name,value:(output/name).write_text(json.dumps(value,indent=2)+'\n')
    write('plan.json',plan)
    np.savez_compressed(output/'prediction.npz',initial_geometry=np.broadcast_to(data['nodes'],(len(data['times']),*data['nodes'].shape)),
                        **{'sealed_'+name:value for name,value in baselines.items()},**{k:data[k]for k in ['nodes','observation_matrix','observed_initial']})
    write('inference-sealed.json',{'plan_sha256':sha(output/'plan.json'),'forecast_target_access':False,'train_end':data['train_end']})
    write('complete.json',{'completed':True,'held_target_access':False,'fit_end':data['train_end'],'frames':len(data['times']),'prediction_sha256':sha(output/'prediction.npz')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args();prepare(a.package,a.output,a.protocol,a.protocol_sha256)
