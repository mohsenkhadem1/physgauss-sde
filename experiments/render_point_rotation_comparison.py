"""Post-seal LOCAL actual-versus-predicted videos for rotation-aware v0.2.

Fixed deterministic illustrations selected after all32 primary results seal;
selection is descriptive, never model/action tuning or a representative sample.
Only A31..60 forecast frames and separate reset-control B0..60 are displayed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
try:
    import summarize_point_rotation_analysis as aggregate
    import render_point_prior_comparison as renderer_v1
except ImportError:
    from experiments import summarize_point_rotation_analysis as aggregate
    from experiments import render_point_prior_comparison as renderer_v1

ROOT=Path(__file__).resolve().parents[1]
SOURCE_NAMES=('render_point_rotation_comparison.py','render_point_prior_comparison.py',
    'summarize_point_rotation_analysis.py','run_point_rotation_analysis.py')
SELECTIONS=('largest_active_improvement','worst_active_regression','worst_active_absolute_error')
RECIPE_SCHEMA='point-rotation-video-recipe-v0.2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(name):raise ValueError('Canonical repository-relative path required')
    path=(ROOT/p).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Repository escape')
    return path


def record(path):return dict(path=Path(path).resolve().relative_to(ROOT.resolve()).as_posix(),sha256=sha(path))
def checked(rec):
    path=resolve(rec['path'])
    if sha(path)!=rec['sha256']:raise ValueError('Frozen render input changed: '+rec['path'])
    return path

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def recipe_choices():
    return dict(selection_metric='A31..60 coordinate MSE, active chosen view versus no additional view',
        selection_roles=list(SELECTIONS),tie_break='Lowest object index',duplicate_case_policy='One video pair per unique case; retain all selection labels',
        A_frame_indices=list(range(31,61)),B_frame_indices=list(range(61)),fps=20,
        pixels=[1536,576],public_rest_bounds_margin_m=.04,motion_magnification=1.,
        camera='Orthographic; elevation22+3sin(pi*t/3), azimuth-65+30*t/3; same across panels, public time only',
        color='viridis of public rest-face x coordinate; identical across panels',clipping_policy='Fail and preserve diagnostic; no adaptive zoom/bounds or deformation magnification')


def prepare_recipe(path,execution_record,method_record,output_prefix):
    target=resolve(path)
    if target.exists():raise ValueError('Existing recipe retained')
    resolve(output_prefix)
    recipe=dict(schema=RECIPE_SCHEMA,execution_protocol=execution_record,method=method_record,output_prefix=output_prefix,
        source_sha256={'experiments/'+name:sha(ROOT/'experiments'/name) for name in SOURCE_NAMES},choices=recipe_choices(),
        require_complete_new32_primary_seal=True,execute_authorization='ROOT_SEPARATE_AFTER_PRIMARY_SEAL',
        new_inference_physics_or_observation=False,raw_video_PNG_LOCAL=True)
    target.parent.mkdir(parents=True,exist_ok=True);write(target,recipe);return record(target)


def validate_recipe(path,digest):
    recipe=read(checked(dict(path=path,sha256=digest)))
    if (recipe.get('schema')!=RECIPE_SCHEMA or recipe.get('choices')!=recipe_choices()
        or recipe.get('require_complete_new32_primary_seal') is not True
        or recipe.get('execute_authorization')!='ROOT_SEPARATE_AFTER_PRIMARY_SEAL'
        or recipe.get('new_inference_physics_or_observation') is not False):raise ValueError('Exact frozen video recipe required')
    sources={'experiments/'+name:sha(ROOT/'experiments'/name) for name in SOURCE_NAMES}
    if recipe.get('source_sha256')!=sources:raise ValueError('Render source closure changed')
    return recipe


def gate_primary(recipe,evaluation_record):
    """Completed aggregate first; all32 sources/states/choices before selection."""
    evaluation=read(checked(evaluation_record));execution_sha=recipe['execution_protocol']['sha256'];method_sha=recipe['method']['sha256']
    if (evaluation.get('schema')!='point-rotation-reserved-view-evaluation-v0.2'
        or evaluation.get('completed') is not True or evaluation.get('objects')!=32
        or evaluation.get('method_sha256')!=method_sha or evaluation.get('execution_protocol_sha256')!=execution_sha):
        raise ValueError('Final completed NEW32 primary aggregate required before selection or arrays')
    release=read(checked(evaluation['release']))
    if (release.get('schema')!=aggregate.RELEASE_SCHEMA or release.get('status')!='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE'
        or release.get('method')!=recipe['method'] or release.get('method_sha256')!=method_sha
        or release.get('execution_protocol_sha256')!=execution_sha or release.get('object_indices')!=list(range(16,48))
        or release.get('calibration_freeze')!=evaluation['calibration_freeze']
        or release.get('no_reserved_response_or_truth_used_for_calibration') is not True):raise ValueError('Exact new frozen reserved release required')
    freeze=aggregate.validate_freeze(evaluation['calibration_freeze'],execution_sha)
    if freeze['execution_protocol']!=recipe['execution_protocol'] or freeze['method']!=recipe['method']:raise ValueError('Calibration/execution/method differs')
    execution,_,plan=aggregate.check_execution(checked(recipe['execution_protocol']),execution_sha)
    status=read(resolve(execution['analysis_output'])/'reserved-execution.json')
    if (status.get('terminal') is not True or status.get('completed') is not True or status.get('role')!='RESERVED_EVALUATION'
        or status.get('method_sha256')!=method_sha or status.get('execution_protocol_sha256')!=execution_sha
        or status.get('planned_object_indices')!=list(range(16,48)) or status.get('completed_object_indices')!=list(range(16,48))
        or status.get('unlaunched_object_indices')!=[] or status.get('failures')!=[]):raise ValueError('All32 primary analysis slots must be terminal without failures')
    objects=aggregate.eligible(plan,'RESERVED_EVALUATION',execution_sha,method_sha,
        release=release,calibration_freeze=freeze,calibration_freeze_record=evaluation['calibration_freeze'])
    values,records=aggregate.load_results(resolve(execution['analysis_output']),objects,'RESERVED_EVALUATION',execution,execution_sha)
    if [o['index'] for o in objects]!=list(range(16,48)) or records!=evaluation['reserved_results']:raise ValueError('All32 exact sealed result records required')
    return evaluation,execution,plan,{v['object_index']:v for v in values}


def select_cases(values):
    """Fixed signed MSE rankings, stable lowest-index ties; never inspect arrays."""
    if sorted(values)!=list(range(16,48)):raise ValueError('Exact32 sealed object metrics required')
    rows=[]
    for i in range(16,48):
        value=values[i];active=value['selected_active_view'];states=value['states']
        no=float(states['no_additional_view']['A_after1p5s_MSE_m2']);yes=float(states[f'fixed_view_{active}']['A_after1p5s_MSE_m2'])
        if type(active) is not int or not 0<=active<4 or not np.isfinite([no,yes]).all() or min(no,yes)<0:raise ValueError('Finite sealed view metrics required')
        rows.append(dict(object_index=i,no_view_MSE_m2=no,active_MSE_m2=yes,improvement_MSE_m2=no-yes,regression_MSE_m2=yes-no))
    key_names=('improvement_MSE_m2','regression_MSE_m2','active_MSE_m2');selected=[]
    for role,key in zip(SELECTIONS,key_names):
        row=max(rows,key=lambda r:(r[key],-r['object_index']))
        selected.append(dict(selection=role,ranking_metric=key,**row))
    return selected


def panel_labels(program):
    if program=='A':
        return 'FUTURE A | frames31..60 (1.55..3.00s)', 'Conditioning A depths:0.5/1.0s; optional view1.2s. No observations during displayed forecast.'
    if program=='B':
        return 'RESET B | known control, frames0..60 (0..3.00s)', 'B starts from rest; no B observations. Material/support prediction is conditioned on preceding A observations only.'
    raise ValueError('Only declared A forecast or independent-reset B control')


def render_sequence(rest,faces,times,trajectories,output,*,program,caption='',fps=20):
    """No exaggeration or adaptive camera; fixed geometry-only synthetic-test API."""
    header,description=panel_labels(program);output=Path(output)
    rest=np.asarray(rest,dtype=np.float64);faces=np.asarray(faces);times=np.asarray(times,dtype=np.float64)
    if output.exists():raise ValueError('Fresh LOCAL video path required')
    if (rest.ndim!=2 or rest.shape[1]!=3 or not np.isfinite(rest).all() or faces.ndim!=2 or faces.shape[1]!=3
        or faces.dtype.kind not in 'iu' or (faces<0).any() or (faces>=len(rest)).any() or not len(faces)
        or times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
        or type(fps) is not int or fps<=0 or len(trajectories)!=3
        or any(q.shape!=(len(times),len(rest),3) or not np.isfinite(q).all() for q in trajectories)):
        raise ValueError('Three finite shape-matched trajectories, mesh and monotone actual timestamps required')
    limits=np.stack([rest.min(0)-.04,rest.max(0)+.04],axis=1)
    clipped=[int(np.max(np.sum(np.any((q<limits[:,0])|(q>limits[:,1]),axis=2),axis=1))) for q in trajectories]
    output.parent.mkdir(parents=True,exist_ok=True)
    if any(clipped):
        write(output.with_suffix('.failure.json'),dict(completed=False,reason='Vertices outside prospectively fixed public-rest bounds',maximum_vertices_outside_fixed_bounds=clipped,limits_m=limits.tolist()))
        raise ValueError('Clipped geometry retained as failure; no adaptive bounds')
    plt=renderer_v1.plt;fig=plt.figure(figsize=(15.36,5.76),dpi=100,facecolor='#f7f9fc')
    names=['Actual · independent FEM','No-extra-view predictive mean','Active-view predictive mean']
    colors=renderer_v1.cm.viridis((rest[faces].mean(1)[:,0]-rest[:,0].min())/max(np.ptp(rest[:,0]),1e-12));panels=[]
    for j,name in enumerate(names):
        ax=fig.add_subplot(1,3,j+1,projection='3d');ax.set_title(name,fontsize=10,pad=0)
        mesh=renderer_v1.Poly3DCollection(rest[faces],facecolors=colors,edgecolors='#405467',linewidths=.35,alpha=.75);ax.add_collection3d(mesh)
        for axis,bounds in zip('xyz',limits):getattr(ax,'set_'+axis+'lim')(bounds);getattr(ax,'set_'+axis+'label')(axis+' (m)',fontsize=8)
        ax.set_box_aspect(limits[:,1]-limits[:,0]);ax.set_proj_type('ortho');ax.tick_params(labelsize=6);ax.grid(False);panels.append((ax,mesh))
    fig.subplots_adjust(left=.01,right=.99,bottom=.13,top=.79,wspace=.01)
    title=fig.text(.5,.96,'',ha='center',fontsize=10);fig.text(.5,.90,header,ha='center',fontsize=10)
    fig.text(.5,.85,description,ha='center',fontsize=8)
    fig.text(.5,.045,'True scale; same fixed public-rest bounds, orbit and colors. Rotation-aware approximate mechanics + fitted discrepancy.',ha='center',fontsize=8)
    fig.text(.5,.015,'Predictive mean geometry only; uncertainty bands are not depicted. No native Isaac or new simulation claim.',ha='center',fontsize=8)
    writer=renderer_v1.imageio_ffmpeg.write_frames(str(output),(1536,576),fps=fps,codec='libx264',quality=7,macro_block_size=2,
        ffmpeg_log_level='error',output_params=['-threads','2','-movflags','+faststart']);writer.send(None)
    try:
        for frame,t in enumerate(times):
            title.set_text(caption+f' | simulation t={t:.2f}s')
            for (ax,mesh),q in zip(panels,trajectories):mesh.set_verts(q[frame,faces]);ax.view_init(elev=22+3*np.sin(np.pi*t/3),azim=-65+30*t/3)
            fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            if frame==0:plt.imsave(output.with_suffix('.first-frame.png'),rgb)
            writer.send(rgb)
    finally:writer.close();plt.close(fig)
    reader=renderer_v1.imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24');info=next(reader);count=sum(1 for _ in reader)
    if count!=len(times) or tuple(info['size'])!=(1536,576) or abs(info['duration']-len(times)/fps)>1/fps:raise ValueError('Encoded frame/size/duration mismatch')
    return dict(output=str(output),sha256=sha(output),bytes=output.stat().st_size,frames=count,fps=fps,duration_s=info['duration'],
        first_simulation_time_s=float(times[0]),last_simulation_time_s=float(times[-1]),program=program,limits_m=limits.tolist(),
        maximum_vertices_outside_fixed_bounds=clipped,motion_magnification=1.,same_camera_scale_colors=True,
        first_frame_png=dict(path=str(output.with_suffix('.first-frame.png')),sha256=sha(output.with_suffix('.first-frame.png'))))


def run(recipe_path,recipe_sha256,evaluation_record):
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    recipe=validate_recipe(recipe_path,recipe_sha256)
    evaluation,execution,plan,values=gate_primary(recipe,evaluation_record)
    selected=select_cases(values);destination=resolve(recipe['output_prefix'])
    if destination.exists():raise ValueError('Fresh LOCAL render directory required')
    destination.mkdir(parents=True)
    write(destination/'selection-frozen.json',dict(evaluation=evaluation_record,selections=selected,
        made_after_primary_seal=True,selection_before_trajectory_or_prediction_array_reads=True,selection_metric=recipe['choices']['selection_metric']))
    status=dict(completed=False,terminal=False,selected_cases=selected,recipe=record(resolve(recipe_path)),raw_arrays_opened=False);write(destination/'execution.json',status)
    rows=[]
    try:
        public=read(checked(plan['public_hardware']));rest=np.asarray(public['rest_points_m'],dtype=np.float64);faces=np.asarray(public['surface_faces'],dtype=np.int64)
        for index in sorted({row['object_index'] for row in selected}):
            value=values[index];directory=resolve(execution['analysis_output'])/'objects'/value['object_id']
            state_meta=read(checked(value['posterior_freeze']));roles=[row['selection'] for row in selected if row['object_index']==index]
            capture=value['capture_pairs'];obj=next(o for o in plan['objects'] if o['index']==index)
            # Require exact declared finite A/B provenance before array access.
            for program in 'AB':
                cell=next(c for c in plan['cells'] if c['object_id']==obj['object_id'] and c['id']==obj['object_id']+'__'+program)
                if checked(capture[program]['trajectory'])!=resolve(cell['output'])/'trajectory.npz' or capture[program]['protocol']!=cell['protocol']:raise ValueError('Truth trajectory must match declared same-object reset program')
                terminal=read(checked(capture[program]['execution']));summary=read(checked(capture[program]['summary']))
                if terminal.get('terminal') is not True or terminal.get('completed') is not True or summary.get('completed') is not True or summary.get('frames')!=61 or summary.get('trajectory_sha256')!=capture[program]['trajectory']['sha256']:raise ValueError('Complete saved finite trajectory required')
            status['raw_arrays_opened']=True;write(destination/'execution.json',status)
            with np.load(checked(state_meta['states']),allow_pickle=False) as raw:
                times=raw['times_s'].copy()
                if not np.array_equal(raw['rest_points_m'],rest) or not np.array_equal(times,np.arange(61)/20):raise ValueError('Saved predictive rest or actual20Hz clock differs')
                means={program:[rest+raw[name+'__'+program+'__mean_displacement_m'] for name in ['no_additional_view',f"fixed_view_{value['selected_active_view']}"]] for program in 'AB'}
            for program in 'AB':
                with np.load(checked(capture[program]['trajectory']),allow_pickle=False) as raw:
                    truth=raw['positions_m'].copy()
                    if not np.array_equal(raw['times_s'],times):raise ValueError('Finite and predictive timestamps differ')
                frames=np.asarray(recipe['choices'][program+'_frame_indices'])
                row=render_sequence(rest,faces,times[frames],[truth[frames],*[q[frames] for q in means[program]]],destination/f'object-{index:03d}-{program}.mp4',program=program,
                    caption=f'Object{index} | '+', '.join(roles),fps=recipe['choices']['fps'])
                for pathkey in ['output']:row[pathkey]=Path(row[pathkey]).resolve().relative_to(ROOT).as_posix()
                row['first_frame_png']['path']=Path(row['first_frame_png']['path']).resolve().relative_to(ROOT).as_posix()
                row.update(object_index=index,selections=roles,active_view_index=value['selected_active_view'],frame_indices=frames.tolist(),
                    coordinate_RMSE_m={name:float(np.sqrt(np.mean((q[frames]-truth[frames])**2))) for name,q in zip(['no_extra_view','active_view'],means[program])},
                    predictive_states=state_meta['states'],finite_trajectory=capture[program]['trajectory'])
                rows.append(row)
        metadata=dict(schema='point-rotation-comparison-videos-v0.2',completed=True,recipe=record(resolve(recipe_path)),sealed_primary_evaluation=evaluation_record,
            selection_frozen=record(destination/'selection-frozen.json'),selection_is_posthoc_descriptive=True,selection_metric=recipe['choices']['selection_metric'],
            selections=selected,videos=rows,raw_MP4_PNG_LOCAL=True,new_inference_physics_or_observations=False,CPU_threads=2,GPU_used=False,
            duration_note='No interpolation or repeated frames: A30 samples at20fps=1.5s; B61inclusive samples at20fps=3.05s.',
            scope='Future A1.55..3s and separate independent-reset known B0..3s. No observations after conditioning cutoffs; no future anticipation claim from retrospective motion.')
        write(destination/'render-manifest.json',metadata);status.update(terminal=True,completed=True,videos=len(rows));write(destination/'execution.json',status)
        return record(destination/'render-manifest.json')
    except BaseException as exc:
        write(destination/'failure.json',dict(completed=False,error=type(exc).__name__,message=str(exc),completed_videos=len(rows)))
        status.update(terminal=True,completed=False,videos=len(rows));write(destination/'execution.json',status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for key in ['output','output-prefix','execution','execution-sha256','method','method-sha256']:p.add_argument('--'+key,required=True)
    p=sub.add_parser('run')
    for key in ['recipe','recipe-sha256','evaluation','evaluation-sha256']:p.add_argument('--'+key,required=True)
    a=parser.parse_args()
    if a.action=='prepare':result=prepare_recipe(a.output,dict(path=a.execution,sha256=a.execution_sha256),dict(path=a.method,sha256=a.method_sha256),a.output_prefix)
    else:result=run(a.recipe,a.recipe_sha256,dict(path=a.evaluation,sha256=a.evaluation_sha256))
    print(json.dumps(result))
