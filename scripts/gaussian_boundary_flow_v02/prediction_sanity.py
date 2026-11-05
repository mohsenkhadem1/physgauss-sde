"""Post-seal consistency diagnostics without any future object/force targets.

Reads predictions, initial geometry and the already sealed prefix package only.
No gate/prediction edits. Floor projection is not physical-accuracy evidence.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest


def _stats(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()))


def audit_arrays(data,plan,complete,pred,contact_pred=None):
    if any(k in data for k in ('evaluation','manual','forces','material','object_points')):
        raise ValueError('Private evaluator/physical-label fields forbidden')
    if len(data['observed'])!=data['train_end']:
        raise ValueError('Prefix-only package required')
    ids=np.asarray(plan['point_ids'],dtype=int)
    frames=np.asarray(plan['prefix_frames'],dtype=int)
    if len(frames)==0 or frames[0]!=0 or np.any(frames<0) or np.any(frames>=data['train_end']):
        raise ValueError('Declared frame0-containing prefix budget required')
    if ids.ndim!=1 or not len(ids) or np.any(ids<0) or np.any(ids>=len(data['observed_initial'])):
        raise ValueError('Invalid declared point budget')
    for key in ('nodes','observation_matrix','observed_initial'):
        if not np.array_equal(pred[key],data[key]):raise ValueError('Changed raw mapping/geometry')
        if contact_pred is not None and not np.array_equal(contact_pred[key],data[key]):
            raise ValueError('Changed derived mapping/geometry')
    mask=np.asarray(data['observed_mask'][0],bool)[ids]
    if not mask.any():raise ValueError('No initial observations within declared point budget')
    A=np.asarray(data['observation_matrix'])[ids][mask];initial=np.asarray(data['observed_initial'])[ids][mask]
    nodes=np.asarray(data['nodes']);T=len(data['times']);floor=float(data['floor'])
    rows={}
    for name in plan['prediction_methods']:
        if not name.endswith('_raw'):continue
        trajectory=np.asarray(pred[name]);family=name[:-4]
        if trajectory.shape!=(T,len(nodes),3):raise ValueError('Full prediction path required')
        selection=complete.get('forecast_samples',{}).get(family)
        if selection is not None:
            selection={k:selection.get(k) for k in ('requested','valid','failures')}
            selection['failed']=len(selection['failures'] or [])
        row=dict(finite=bool(np.isfinite(trajectory).all()),
                 surviving_draws=selection,
                 frames=T,initial_points=int(mask.sum()))
        def measure(paths):
            mapped=initial[None]+np.einsum('pn,snc->spc',A,paths[:,0]-nodes[None])
            initial_rms=np.sqrt(np.mean((mapped-initial[None])**2,axis=(1,2)))*1000
            penetration=np.maximum(0,floor-paths[:,:,:,2])*1000
            return dict(initial_observation_coordinate_rmse_mm=_stats(initial_rms),
                        allframe_maximum_floor_penetration_mm=float(penetration.max()),
                        draw_fraction_any_penetration=float((penetration.max(axis=(1,2))>0).mean()),
                        floor_provenance='Provided initial min-z plane; may not equal a measured scene surface')
        if row['finite']:row['mean_prediction']=measure(trajectory[None])
        samples=pred.get(name+'_samples')
        if samples is not None:
            samples=np.asarray(samples);row['stored_draws']=len(samples)
            if samples.ndim!=4 or samples.shape[1:]!=trajectory.shape:raise ValueError('Invalid raw sample path')
            if np.isfinite(samples).all():row['sample_diagnostics']=measure(samples)
            else:row['sample_diagnostics']={'available':False,'reason':'Nonfinite inherited sample; no silent filtering'}
        if contact_pred is not None:
            candidates=[n for n in (family+'_pose',family+'_contact') if n in contact_pred]
            if len(candidates)!=1:raise ValueError('Exactly one declared corresponding corrected channel required')
            corrected=np.asarray(contact_pred[candidates[0]])
            if corrected.shape!=trajectory.shape:raise ValueError('Changed corrected trajectory clock')
            if row['finite'] and np.isfinite(corrected).all():
                change=corrected-trajectory
                row['pose_contact_correction']=dict(
                    coordinate_rms_mm=float(np.sqrt(np.mean(change**2))*1000),
                    maximum_node_translation_mm=float(np.linalg.norm(change,axis=-1).max()*1000),
                    mean_frame_coordinate_rms_mm=float(np.sqrt(np.mean(change**2,axis=(1,2))).mean()*1000),
                    corrected_floor_penetration_mm=float(np.maximum(0,floor-corrected[:,:,2]).max()*1000),
                    interpretation='Rigid/kinematic modification magnitude, not elastic force consistency')
            else:row['pose_contact_correction']={'available':False,'reason':'Nonfinite inherited prediction retained'}
        rows[name]=row
    return dict(case=data.get('name'),scope='Post-seal target-free consistency audit; does not change predictions or admission gates',
                methods=rows,point_ids=ids.tolist(),prefix_frames=frames.tolist(),
                caveats=['Relative-feature subtraction cancels arbitrary initial offsets; small relative error does not imply a correct initial force-balanced state',
                         'Floor projection can enforce nonpenetration without physical force or energy accuracy',
                         'Surviving-draw diagnostics exclude inherited failed solves and are not unconditional calibrated uncertainty',
                         'No private force/material labels or future object observations are read'],future_target_access=False)


def audit_run(run,contact_run=None):
    run=Path(run);plan,seal,complete,pred=load_sealed(run)
    package=Path(plan['input_package'])
    if sha(package/'manifest.json')!=plan['input_package_manifest_sha256']:raise ValueError('Changed package manifest')
    data=load_package(package)
    if predictor_digest(data)!=plan['input_digest']:raise ValueError('Changed sealed predictor')
    derived=None
    if contact_run is not None:
        cplan,cseal,ccomplete,derived=load_sealed(contact_run)
        if cplan.get('parent_sha256',{}).get('prediction.npz')!=sha(run/'prediction.npz'):
            raise ValueError('Corrected prediction belongs to different parent')
        for key in ('case','input_digest','prefix_frames','point_ids'):
            if cplan[key]!=plan[key]:raise ValueError('Changed corrected observation budget/identity')
    report=audit_arrays(data,plan,complete,pred,derived)
    report['source_sha256']=dict(prediction=sha(run/'prediction.npz'),plan=sha(run/'plan.json'),audit_source=sha(__file__))
    if contact_run is not None:report['source_sha256']['corrected_prediction']=sha(Path(contact_run)/'prediction.npz')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--contact-run');parser.add_argument('--output',required=True);args=parser.parse_args()
    report=audit_run(args.run,args.contact_run);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Do not overwrite an existing sanity audit')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
