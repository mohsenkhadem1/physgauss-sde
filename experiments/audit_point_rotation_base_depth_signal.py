"""All32 acquired72-ray contrast: FEM surface versus best volume-rigid fit.

No likelihood, noise sampling, inference or model tuning. Differences are only
on paired-visible noiseless camera-axis depths; censored channels stay absent.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
try:
    import audit_point_rotation_deformation as geometry
    from campaign_sparse_depth_queries import predict_ranges
    from censored_range_observation import CensoredRanges
except ImportError:
    from experiments import audit_point_rotation_deformation as geometry
    from experiments.campaign_sparse_depth_queries import predict_ranges
    from experiments.censored_range_observation import CensoredRanges
ROOT=geometry.ROOT
resolve,checked,record,sha,read,write=geometry.resolve,geometry.checked,geometry.record,geometry.sha,geometry.read,geometry.write


def paired_depth_metrics(actual,rigid,sigma_m=.0002):
    """One observed/query subset; never fabricate a residual for a missed ray."""
    if not isinstance(actual,CensoredRanges) or not isinstance(rigid,CensoredRanges) or actual.visible.ndim!=1 or actual.visible.shape!=rigid.visible.shape or not len(actual.visible):raise ValueError('Matched nonempty one-dimensional censored predictions required')
    if not np.isfinite(sigma_m) or sigma_m<=0:raise ValueError('Positive single-observation noise reference required')
    a,b=actual.visible,rigid.visible;both=a&b;delta=actual.residual_m[both]-rigid.residual_m[both];n=int(both.sum())
    rms=float(np.sqrt(np.mean(delta*delta))) if n else None
    return dict(rays=len(a),both_visible_rays=n,true_visible_rigid_miss_rays=int(np.sum(a&~b)),true_miss_rigid_visible_rays=int(np.sum(~a&b)),
        both_background_rays=int(np.sum(~a&~b)),visibility_changed_rays=int(np.sum(a!=b)),
        true_visible_rays=int(a.sum()),rigid_visible_rays=int(b.sum()),paired_depth_squared_difference_sum_m2=float(np.sum(delta*delta)),
        paired_visible_depth_difference_RMS_m=rms,paired_visible_depth_difference_bias_m=float(np.mean(delta)) if n else None,
        maximum_absolute_paired_visible_difference_m=float(np.max(np.abs(delta))) if n else None,
        paired_visible_RMS_over_single_observation_sigma=rms/sigma_m if n else None,
        paired_visible_rays_absolute_contrast_above_sigma=int(np.sum(np.abs(delta)>sigma_m)))


def population_summary(objects):
    if [o['object_index'] for o in objects]!=list(range(16,48)):raise ValueError('All32 complete objects required; no filtering')
    indices=np.random.default_rng(2026120900).integers(0,32,size=(10000,32));out={}
    for window in ['base72','frame10','frame20']:
        rows=[o[window] for o in objects];n=sum(r['both_visible_rays'] for r in rows);sse=sum(r['paired_depth_squared_difference_sum_m2'] for r in rows)
        pooled=float(np.sqrt(sse/n)) if n else None;missing=sum(r['both_visible_rays']==0 for r in rows)
        values=np.array([r['paired_visible_depth_difference_RMS_m'] for r in rows],dtype=float)
        metric=None if missing else dict(mean=float(values.mean()),median=float(np.median(values)),quantile05=float(np.quantile(values,.05)),quantile95=float(np.quantile(values,.95)),minimum=float(values.min()),maximum=float(values.max()),
            object_bootstrap_mean_95_interval=np.quantile(values[indices].mean(1),[.025,.975]).tolist(),
            objects_RMS_above_sigma=int(np.sum(values>.0002)),mean_over_single_observation_sigma=float(values.mean()/.0002))
        out[window]=dict(objects=32,objects_without_paired_visible_rays=missing,per_object_paired_RMS_m=metric,
            pooled_paired_visible_RMS_m=pooled,pooled_RMS_over_single_observation_sigma=pooled/.0002 if n else None,
            maximum_absolute_paired_difference_m=max((r['maximum_absolute_paired_visible_difference_m'] for r in rows if r['maximum_absolute_paired_visible_difference_m'] is not None),default=None),
            total_counts={key:sum(r[key] for r in rows) for key in ['rays','both_visible_rays','true_visible_rigid_miss_rays','true_miss_rigid_visible_rays','both_background_rays','visibility_changed_rays','true_visible_rays','rigid_visible_rays','paired_visible_rays_absolute_contrast_above_sigma']},
            objects_with_any_visibility_change=sum(r['visibility_changed_rays']>0 for r in rows))
    return out


def prepare(path,population_protocol_record,population_summary_record,output_prefix):
    target=resolve(path)
    if target.exists():raise ValueError('Existing diagnostic declaration retained')
    prior=read(checked(population_protocol_record));completed=read(checked(population_summary_record))
    if completed.get('completed') is not True or completed.get('objects')!=32 or completed['protocol']!=population_protocol_record:raise ValueError('Completed frozen all32 geometry audit required')
    execution=read(checked(prior['execution_protocol']));plan=read(checked(prior['cohort_plan']))
    sources=dict(execution['source_sha256'])
    for name in ['audit_point_rotation_deformation.py','audit_point_rotation_base_depth_signal.py','campaign_sparse_depth_queries.py','campaign_ray_sensor.py','censored_range_observation.py','render_independent_trajectory.py']:
        rel='experiments/'+name;value=sha(ROOT/rel)
        if rel in sources and sources[rel]!=value:raise ValueError('Conflicting frozen runtime source')
        sources[rel]=value
    p=dict(schema='point-rotation-base-depth-signal-audit-v0.1',status='FROZEN_RETROSPECTIVE_NO_INFERENCE_OR_MODEL_TUNING',
        population_protocol=population_protocol_record,population_summary=population_summary_record,primary_seal=prior['sealed_primary'],
        public_hardware=prior['public_hardware'],public_queries=plan['public_acquired_queries'],cases=[c for c in prior['cases'] if c['program']=='A'],
        source_sha256=sources,output_prefix=output_prefix,device='cpu',CPU_threads=2,GPU=False,wall_limit_s=180,
        choices=dict(object_indices=list(range(16,48)),frame_indices=[10,20],rays_per_frame=36,total_base_rays=72,
            comparison='True saved FEM surface against per-frame best proper volume-lumped rigid fit to same rest surface',
            depth_semantics='Unnormalized camera rays: camera-axis depth in metres, not Euclidean ray distance',
            paired_foreground_only=True,visibility_changes_separate=True,sigma_single_observation_m=.0002,noise_draws=False,
            foreground_contrast='Difference of two noiseless surfaces divided by one observation sigma, not sqrt(2)*sigma',
            bootstrap_replicates=10000,bootstrap_seed=2026120900,bootstrap_unit='Complete object with paired10/20queries',
            absent_paired_hits='Retain object and visibility counts; per-object mean/CI undefined if any object lacks paired hits, no filtering',
            claim='Geometric contrast at the fixed acquired pixels; not full inference information, Fisher information or material identifiability'),
        material_support_labels_read=False,new_physics=False,per_object_ray_statistics_LOCAL=True)
    target.parent.mkdir(parents=True,exist_ok=True);write(target,p);return record(target)


def run(protocol_path,digest):
    p=read(checked(dict(path=protocol_path,sha256=digest)))
    if p.get('schema')!='point-rotation-base-depth-signal-audit-v0.1' or p.get('device')!='cpu' or p.get('CPU_threads')!=2 or p.get('GPU') is not False or [c['object_index'] for c in p['cases']]!=list(range(16,48)):raise ValueError('Exact fixed all32 CPU diagnostic required')
    for name,value in p['source_sha256'].items():checked(dict(path=name,sha256=value))
    if p['source_sha256'].get(Path(__file__).relative_to(ROOT).as_posix())!=sha(__file__):raise ValueError('Own source not frozen')
    for key in ['population_protocol','population_summary','primary_seal']:checked(p[key])
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    import torch
    torch.set_num_threads(2)
    public=read(checked(p['public_hardware']));queries=read(checked(p['public_queries']))
    if len(queries)!=2 or [q['frame_index'] for q in queries]!=[10,20] or any(len(q['xs'])!=36 for q in queries):raise ValueError('Exact existing72 acquired rays atA10/20 required')
    rest=np.asarray(public['rest_points_m'],dtype=np.float64);tets=np.asarray(public['tetrahedra'],dtype=np.int64);faces=np.asarray(public['surface_faces'],dtype=np.int64)
    _,_,weights=geometry.reference_geometry(rest,tets);output=resolve(p['output_prefix'])
    if output.exists():raise ValueError('Existing diagnostic retained; no retry')
    output.mkdir(parents=True);start=time.monotonic();status=dict(terminal=False,completed=False,completed_objects=[],GPU_used=False,labels_read=False);write(output/'execution.json',status);objects=[]
    try:
        for case in p['cases']:
            if time.monotonic()-start>p['wall_limit_s']:raise TimeoutError('Bounded180s depth diagnostic wall limit')
            for key in ['summary','execution','protocol']:checked(case[key])
            with np.load(checked(case['trajectory']),allow_pickle=False) as z:q=z['positions_m'].copy();times=z['times_s'].copy()
            if q.dtype!=np.float64 or q.shape!=(61,len(rest),3) or not np.array_equal(times,np.arange(61)/20):raise ValueError('All61 float64 saved frames required')
            fit=geometry.rigid_fit(rest,q,weights);rigid=np.einsum('nc,tdc->tnd',rest,fit['rotation_world_from_rest'])+fit['translation_m'][:,None]
            with torch.no_grad():pred=predict_ranges(np.stack([q,rigid]),faces,queries,device='cpu',particle_chunk=2)
            actual=CensoredRanges(pred.visible[0],pred.residual_m[0]);counterfactual=CensoredRanges(pred.visible[1],pred.residual_m[1])
            row=dict(object_index=case['object_index'],trajectory=case['trajectory'],base72=paired_depth_metrics(actual,counterfactual),
                actual_visible=actual.visible.tolist(),rigid_visible=counterfactual.visible.tolist(),
                paired_only_depth_difference_m=np.where(actual.visible&counterfactual.visible,actual.residual_m-counterfactual.residual_m,np.nan).tolist())
            # JSON omits absent differences as null, never a fabricated zero depth.
            row['paired_only_depth_difference_m']=[None if not np.isfinite(v) else v for v in row['paired_only_depth_difference_m']]
            for j,frame in enumerate([10,20]):
                sl=slice(j*36,(j+1)*36);row['frame'+str(frame)]=paired_depth_metrics(CensoredRanges(actual.visible[sl],actual.residual_m[sl]),CensoredRanges(counterfactual.visible[sl],counterfactual.residual_m[sl]))
            objects.append(row);status['completed_objects'].append(case['object_index']);status['elapsed_s']=time.monotonic()-start;write(output/'execution.json',status)
        write(output/'per-object-ray-statistics-LOCAL.json',objects)
        summary=dict(schema='point-rotation-base-depth-signal-summary-v0.1',completed=True,objects=32,program='A',acquired_frames=[10,20],base_channels_per_object=72,
            protocol=record(resolve(protocol_path)),primary_seal=p['primary_seal'],public_queries=p['public_queries'],public_hardware=p['public_hardware'],
            population=population_summary(objects),sensor_sigma_m=.0002,depth_semantics=p['choices']['depth_semantics'],
            interpretation='Actual noiseless FEM versus best volume-weighted proper-rigid fit at exact base sensor pixels; no invented missed-ray measurements or noise draws.',
            limitations=['Foreground RMS contrasts only use paired-visible rays; changed visibility is counted separately.','Counterfactual rigid fit is a descriptive geometric projection, not a support-admissible physical model.',
                'Depth contrast/noise ratio is not total multi-ray information, material identifiability or posterior accuracy.','Only acquiredA0.5/1.0s72channels are tested; no future/newview observations.'],
            independent_unit='Complete object;10,000 object bootstrap means, seed2026120900',GPU_used=False,labels_read=False,new_physics_or_inference=False,elapsed_s=time.monotonic()-start)
        write(output/'summary.json',summary);status.update(terminal=True,completed=True,elapsed_s=time.monotonic()-start);write(output/'execution.json',status);return record(output/'summary.json')
    except BaseException as exc:
        write(output/'per-object-ray-statistics-LOCAL-PARTIAL.json',objects);write(output/'failure.json',dict(completed=False,error=type(exc).__name__,message=str(exc)))
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-start);write(output/'execution.json',status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for key in ['output','output-prefix','population-protocol','population-protocol-sha256','population-summary','population-summary-sha256']:p.add_argument('--'+key,required=True)
    p=sub.add_parser('run');p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='prepare':value=prepare(a.output,dict(path=a.population_protocol,sha256=a.population_protocol_sha256),dict(path=a.population_summary,sha256=a.population_summary_sha256),a.output_prefix)
    else:value=run(a.protocol,a.protocol_sha256)
    print(json.dumps(value))
