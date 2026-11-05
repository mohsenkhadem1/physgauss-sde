"""Retrospective rigid controls for the original four sealed PhysTwin cases.

The frozen expansion baseline allowlist is unchanged. This wrapper reuses its
pure prediction helpers and takes exact observation IDs from each parent plan.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02 import phystwin_expansion_baseline as baseline
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed
from scripts.gaussian_boundary_flow_v02.freeze_validation import dependencies

CASES=('single_push_sloth','single_lift_sloth','single_lift_zebra','double_lift_zebra')


def parent(case):return Path('runs/gaussian-boundary-flow-v03-package'if case=='single_push_sloth'else'runs/gaussian-boundary-flow-v03-expansion')/case

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def numeric_controls(data,frames,ids):
    initial=np.broadcast_to(data['nodes'],(len(data['times']),*data['nodes'].shape)).copy()
    controls=baseline.prefix_baselines(data,frames,ids)
    try:worlds,meta=baseline.predict(data,initial[None],prefix_frames=frames,point_ids=ids);rigid=worlds[0]
    except (ValueError,RuntimeError,np.linalg.LinAlgError)as error:rigid=np.full_like(initial,np.nan);meta={'failed':True,'error':str(error),'policy':baseline.POLICY}
    return initial,rigid,controls,meta


def freeze(path):
    if Path(path).exists():raise FileExistsError('Retrospective control manifest already frozen')
    jobs=[]
    for case in CASES:
        source=parent(case);plan,_,_,_=load_sealed(source);package=Path(plan['input_package'])
        if baseline.sha(package/'manifest.json')!=plan['input_package_manifest_sha256']:raise ValueError('Parent package changed')
        data=baseline.load_package(package)
        if baseline.predictor_digest(data)!=plan['input_digest']:raise ValueError('Parent input digest changed')
        jobs.append({'case':case,'parent':str(source),'parent_plan_sha256':baseline.sha(source/'plan.json'),'parent_prediction_sha256':baseline.sha(source/'prediction.npz'),
                     'package':str(package),'package_manifest_sha256':baseline.sha(package/'manifest.json'),'input_digest':plan['input_digest'],
                     'nodes':len(data['nodes']),'prefix_frames':plan['prefix_frames'],'point_ids':plan['point_ids'],
                     'output':str(Path('runs/gaussian-boundary-flow-v05-original-phys-controls')/case)})
    manifest={'status':'frozen','scope':'Separate retrospective controls for four historically exposed PhysTwin cases; no refitting or target scoring',
              'reason':'The original frozen baseline CLI accepts only the four new cases. This isolated wrapper keeps case identities and reuses identical pure prediction helpers.',
              'cases':jobs,'source_sha256':dependencies([__file__]),'reference_baseline_source_sha256':baseline.sha(baseline.__file__),
              'recipe':'Repeat initial nodes, apply unchanged shared pose/contact policy with exact parent 12 frames and 48 point IDs; preserve persistence/velocity even if rigid policy fails.',
              'device':'cpu','target_access':False}
    write(path,manifest)


def prepare(job,manifest_digest):
    source=Path(job['parent']);plan,seal,complete,arrays=load_sealed(source)
    if baseline.sha(source/'plan.json')!=job['parent_plan_sha256']or baseline.sha(source/'prediction.npz')!=job['parent_prediction_sha256']:raise ValueError('Parent seal changed')
    package=Path(job['package'])
    if baseline.sha(package/'manifest.json')!=job['package_manifest_sha256']:raise ValueError('Package changed')
    data=baseline.load_package(package)
    if data['name']!=job['case']or baseline.predictor_digest(data)!=job['input_digest']or plan['prefix_frames']!=job['prefix_frames']or plan['point_ids']!=job['point_ids']:raise ValueError('Case or observation identity changed')
    for key in ('nodes','observed_initial','observation_matrix'):
        if not np.array_equal(data[key],arrays[key]):raise ValueError('Parent geometry changed')
    out=Path(job['output'])
    if out.exists():raise FileExistsError('Control output is immutable')
    out.mkdir(parents=True)
    derived=dict(plan,prediction_methods=['initial_geometry','rigid_contact'],scope='Retrospective initial shape and shared rigid pose/contact controls; no mechanical parameter inference or calibrated uncertainty',
                 control_manifest_sha256=manifest_digest,reference_baseline_source_sha256=baseline.sha(baseline.__file__),parent_prediction_sha256=job['parent_prediction_sha256'])
    write(out/'plan.json',derived)
    initial,rigid,controls,metadata=numeric_controls(data,job['prefix_frames'],job['point_ids'])
    np.savez_compressed(out/'prediction.npz',initial_geometry=initial,rigid_contact=rigid,**{'sealed_'+k:v for k,v in controls.items()},**{k:data[k]for k in ('nodes','observed_initial','observation_matrix')})
    write(out/'policy-metadata.json',metadata)
    write(out/'inference-sealed.json',{'plan_sha256':baseline.sha(out/'plan.json'),'train_end':seal['train_end'],'forecast_target_access':False,'parent_prediction_sha256':job['parent_prediction_sha256']})
    write(out/'complete.json',{'completed':True,'held_target_access':False,'fit_end':seal['train_end'],'frames':complete['frames'],'prediction_sha256':baseline.sha(out/'prediction.npz'),'rigid_contact_failed':bool(metadata.get('failed',False))})
    load_sealed(out)
    return {'case':job['case'],'output':str(out),'nodes':len(data['nodes']),'prediction_sha256':baseline.sha(out/'prediction.npz'),'rigid_contact_failed':bool(metadata.get('failed',False))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','predict'],required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256');a=p.parse_args()
    if a.phase=='freeze':freeze(a.manifest);print(baseline.sha(a.manifest))
    else:
        if baseline.sha(a.manifest)!=a.manifest_sha256:raise ValueError('Manifest digest mismatch')
        manifest=json.loads(Path(a.manifest).read_text())
        for source,digest in manifest['source_sha256'].items():
            if baseline.sha(source)!=digest:raise ValueError('Frozen source changed: '+source)
        rows=[]
        for job in manifest['cases']:
            row=prepare(job,a.manifest_sha256);rows.append(row);print(row['case'],row['nodes'],'sealed',flush=True)
        write(Path(a.manifest).with_name('original-phys-controls-complete.json'),{'manifest_sha256':a.manifest_sha256,'cases':rows,'all_four_sealed':True,'target_access':False})
