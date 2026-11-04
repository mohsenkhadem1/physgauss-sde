"""Apply the identical frozen pose/contact policy to prefix-only library retrieval.

No new world selection, suffix fitting, learned-weight edits or posterior claims.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.pose_shape_contact import predict,POLICY


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('Immutable derived run exists')
    original,seal,complete,arrays=load_sealed(source)
    package=Path(original['input_package'])
    if sha(package/'manifest.json')!=original['input_package_manifest_sha256']:raise ValueError('Changed input package')
    data=load_package(package)
    if predictor_digest(data)!=original['input_digest']:raise ValueError('Changed predictor digest')
    if 'retrieval'not in original:raise ValueError('Source must be frozen library retrieval')
    for key in ['nodes','observation_matrix','observed_initial']:
        if not np.array_equal(data[key],arrays[key]):raise ValueError('Changed predictor geometry')
    methods=['nearest_contact']+(['top8_contact']if 'retrieval_worlds'in arrays else [])
    plan={**original,'prediction_methods':methods,'scope':'Library retrieval with exactly the learned comparator frozen pose/contact policy; no posterior uncertainty claim',
          'derived_policy':POLICY,'source_run':str(source.resolve()),
          'parent_sha256':{name:sha(source/name)for name in ['plan.json','inference-sealed.json','complete.json','prediction.npz']},
          'derived_source_sha256':{name:sha(name)for name in [__file__,'scripts/gaussian_boundary_flow_v02/pose_shape_contact.py','scripts/gaussian_boundary_flow_v02/pose_shape.py','scripts/contact_benchmark_v12/contact_rollout.py','scripts/contact_benchmark_v12/scene.py']}}
    output.mkdir(parents=True);write(output/'plan.json',plan)
    result={key:arrays[key]for key in ['nodes','observation_matrix','observed_initial']};metadata={}
    for name,worlds in [('nearest_contact',arrays['nearest_raw'][None])]+([('top8_contact',arrays['retrieval_worlds'])]if 'retrieval_worlds'in arrays else []):
        if not np.isfinite(worlds).all():
            result[name]=np.full_like(arrays['nearest_raw'],np.nan)
            metadata[name]={'failed':True,'reason':'Inherited failed frozen world; no dropping or replacement'};continue
        try:
            derived,info=predict(data,worlds,prefix_frames=plan['prefix_frames'],point_ids=plan['point_ids'])
            result[name]=derived.mean(0);metadata[name]=info
        except (ValueError,RuntimeError,np.linalg.LinAlgError)as error:
            result[name]=np.full_like(arrays['nearest_raw'],np.nan);metadata[name]={'failed':True,'reason':str(error)}
    np.savez_compressed(output/'prediction.npz',**result);write(output/'pose-policy-metadata.json',metadata)
    write(output/'inference-sealed.json',{'plan_sha256':sha(output/'plan.json'),'forecast_target_access':False,'train_end':seal['train_end'],'derived_from_prediction_sha256':sha(source/'prediction.npz')})
    write(output/'complete.json',{'completed':True,'held_target_access':False,'fit_end':complete['fit_end'],'frames':complete['frames'],
          'prediction_sha256':sha(output/'prediction.npz'),'no_new_sample_filtering':True,'uncertainty':'None; nearest/top8 retrieval not posterior samples','policy_metadata_sha256':sha(output/'pose-policy-metadata.json')})
    load_sealed(output)
    return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.source,a.output)
