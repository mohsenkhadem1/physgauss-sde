"""Mechanics-free baselines for four historically exposed PhysTwin plush cases."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.object_score import prefix_baselines,sha,load_sealed
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.pose_shape_contact import predict,POLICY
from scripts.external_benchmark.data import fps

CASES=('single_lift_dinosor','double_lift_sloth','double_stretch_sloth','double_stretch_zebra')

def prepare(package,output):
    package,output=Path(package),Path(output);data=load_package(package)
    if data['dataset']!='PhysTwin'or data['name']not in CASES:raise ValueError('Unsupported expansion case')
    if output.exists():raise FileExistsError('Baseline output is immutable')
    frames=np.unique(np.linspace(0,data['train_end']-1,12,dtype=int));ids=fps(data['observed_initial'],48)
    baseline=prefix_baselines(data,frames,ids);initial=np.broadcast_to(data['nodes'],(len(data['times']),*data['nodes'].shape)).copy()
    output.mkdir(parents=True)
    write=lambda name,value:(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    plan={'case':data['name'],'dataset':'PhysTwin','args':{'nodes':len(data['nodes'])},'input_package':str(package.resolve()),'input_package_manifest_sha256':sha(package/'manifest.json'),
          'input_digest':predictor_digest(data),'prefix_frames':frames.tolist(),'point_ids':ids.tolist(),'prediction_methods':['initial_geometry','rigid_contact'],
          'scope':'Retrospective, mechanics-free initial shape and rigid pose/contact baselines. Persistence and velocity use the same declared prefix observations. No physical inference or calibrated uncertainty claim.',
          'pose_contact_policy':POLICY,'source_sha256':{__file__:sha(__file__)}}
    write('plan.json',plan)
    try:worlds,metadata=predict(data,initial[None],prefix_frames=frames,point_ids=ids);rigid=worlds[0]
    except (ValueError,RuntimeError,np.linalg.LinAlgError)as error:rigid=np.full_like(initial,np.nan);metadata={'failed':True,'error':str(error),'policy':POLICY}
    np.savez_compressed(output/'prediction.npz',initial_geometry=initial,rigid_contact=rigid,**{'sealed_'+k:v for k,v in baseline.items()},**{k:data[k]for k in ['nodes','observation_matrix','observed_initial']})
    write('policy-metadata.json',metadata)
    write('inference-sealed.json',{'plan_sha256':sha(output/'plan.json'),'train_end':data['train_end'],'forecast_target_access':False})
    write('complete.json',{'completed':True,'fit_end':data['train_end'],'frames':len(data['times']),'held_target_access':False,'prediction_sha256':sha(output/'prediction.npz'),'rigid_contact_failed':bool(metadata.get('failed',False))})
    load_sealed(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.package,a.output)
