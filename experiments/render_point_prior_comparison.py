"""LOCAL three-panel reserved-case video, unlocked by a final metadata seal.

No inference, physics, GPU, private parameter labels or fitting is needed.
Frames before acquisition show retrospective means, not causal forecasts.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
METHOD_SHA='ecbc02153336b36ddef33f0c6c01370b012f3ddbeb1d76bd5c71e15145dc3bff'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=path:raise ValueError('Canonical repository-relative path required')
    p=(ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Repository escape')
    return p


def checked(record):
    p=resolve(record['path'])
    if sha(p)!=record['sha256']:raise ValueError('Frozen render input changed: '+record['path'])
    return p


def gate(evaluation_path,evaluation_sha):
    """Final aggregate/release before labels; all32 freezes before array loads."""
    evaluation=read(checked(dict(path=evaluation_path,sha256=evaluation_sha)))
    if evaluation.get('schema')!='point-prior-reserved-view-evaluation-v0.1' or evaluation.get('completed') is not True or evaluation.get('objects')!=32:
        raise ValueError('Final complete32 reserved aggregate required')
    release=read(checked(evaluation['release']))
    if (evaluation.get('schema')!='point-prior-reserved-view-evaluation-v0.1' or evaluation.get('completed') is not True
            or evaluation.get('objects')!=32 or evaluation.get('method_sha256')!=METHOD_SHA
            or release.get('schema')!='point-prior-reserved-release-v0.1'
            or release.get('status')!='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE'
            or release.get('object_indices')!=list(range(16,48))
            or release.get('execution_protocol_sha256')!=evaluation.get('execution_protocol_sha256')
            or release.get('no_reserved_response_or_truth_used_for_calibration') is not True
            or release.get('method_sha256')!=METHOD_SHA):raise ValueError('Complete reserved aggregate/release required')
    try:from summarize_point_prior_analysis import validate_freeze,check_execution
    except ImportError:from experiments.summarize_point_prior_analysis import validate_freeze,check_execution
    freeze=validate_freeze(release['calibration_freeze'],evaluation['execution_protocol_sha256'])
    if (freeze['execution_protocol']['sha256']!=evaluation['execution_protocol_sha256']
            or evaluation['calibration_freeze']!=release['calibration_freeze']):raise ValueError('Execution/calibration provenance differs')
    execution=check_execution(freeze['execution_protocol']['path'],freeze['execution_protocol']['sha256'])
    base=resolve(execution['analysis_output'])
    expected=[(base/'objects'/f'point_prior_{i:03d}'/'result.json').relative_to(ROOT).as_posix() for i in range(16,48)]
    if [r['path'] for r in evaluation['reserved_results']]!=expected:raise ValueError('All32 exact sealed result records required')
    # Completed test aggregate now authorizes these hash-bound result JSONs,
    # including their parameter-label metrics. No trajectory/state arrays yet.
    frozen={};by_index={}
    for index,record in zip(range(16,48),evaluation['reserved_results']):
        value=read(checked(record));directory=base/'objects'/f'point_prior_{index:03d}'
        if (value.get('object_index')!=index or value.get('role')!='RESERVED_EVALUATION'
                or value.get('completed') is not True or value.get('method_sha256')!=METHOD_SHA
                or value.get('execution_protocol_sha256')!=evaluation['execution_protocol_sha256']
                or value.get('posterior_frozen_before_future_truth') is not True):raise ValueError('Incomplete reserved object result')
        freeze_record=value['posterior_freeze']
        if resolve(freeze_record['path'])!=directory/'posterior-states-frozen.json':raise ValueError('State freeze path/identity differs')
        state=read(checked(freeze_record))
        if (state.get('object_index')!=index or state.get('role')!='RESERVED_EVALUATION'
                or state.get('method_sha256')!=METHOD_SHA or state.get('execution_protocol_sha256')!=evaluation['execution_protocol_sha256']
                or state.get('future_A_or_B_truth_opened') is not False
                or state.get('material_or_support_truth_opened') is not False
                or state.get('source_sha256')!=execution['source_sha256']
                or state.get('state_names')!=['no_additional_view',*[f'fixed_view_{j}' for j in range(4)]]):raise ValueError('Reserved posterior was not completely frozen before targets')
        if resolve(state['states']['path'])!=directory/'predictive-states.npz' or resolve(state['active_choice']['path'])!=directory/'active-choice-frozen.json':raise ValueError('Frozen state/choice path differs')
        checked(state['states']);choice=read(checked(state['active_choice'])) # metadata only, not arrays
        if (value.get('active_choice')!=state['active_choice'] or type(value.get('selected_active_view')) is not int
                or value['selected_active_view'] not in range(4) or choice['selected_index']!=value['selected_active_view']):raise ValueError('Active choice/result differs')
        frozen[index]=(freeze_record,state);by_index[index]=value
    # All32 source/state/choice closures precede any predictive/finite array load.
    return evaluation,execution,frozen,by_index


def render_sequence(rest,faces,times,trajectories,output,*,caption='',fps=20):
    """Geometry-only plotting function also exercised by a synthetic toy test."""
    output=Path(output)
    if output.exists():raise ValueError('Fresh LOCAL video path required')
    if len(trajectories)!=3 or any(q.shape!=(len(times),len(rest),3) or not np.isfinite(q).all() for q in trajectories):raise ValueError('Three finite shape-matched trajectories required')
    output.parent.mkdir(parents=True,exist_ok=True)
    limits=np.stack([rest.min(0)-.04,rest.max(0)+.04],axis=1)
    fig=plt.figure(figsize=(15.36,5.76),dpi=100,facecolor='#f7f9fc')
    names=['Actual · Independent FEM','No-extra-view posterior mean · obs through 1.0s','Active-view posterior mean · obs through 1.2s']
    colors=cm.viridis((rest[faces].mean(1)[:,0]-rest[:,0].min())/max(np.ptp(rest[:,0]),1e-12))
    panels=[]
    for j,name in enumerate(names):
        ax=fig.add_subplot(1,3,j+1,projection='3d');ax.set_title(name,fontsize=10,pad=0)
        mesh=Poly3DCollection(rest[faces],facecolors=colors,edgecolors='#405467',linewidths=.35,alpha=.75)
        ax.add_collection3d(mesh)
        for axis,bounds in zip('xyz',limits):getattr(ax,'set_'+axis+'lim')(bounds);getattr(ax,'set_'+axis+'label')(axis+' (m)',fontsize=8)
        ax.set_box_aspect(limits[:,1]-limits[:,0]);ax.set_proj_type('ortho');ax.tick_params(labelsize=6);ax.grid(False)
        panels.append((ax,mesh))
    fig.subplots_adjust(left=.01,right=.99,bottom=.13,top=.81,wspace=.01)
    title=fig.text(.5,.95,'',ha='center',fontsize=11)
    fig.text(.5,.89,'Early depth observations:0.5/1.0s | extra virtual view:1.2s | future evaluation:t>1.5s',ha='center',fontsize=9)
    fig.text(.5,.045,'True scale, common public-rest bounds/camera/colors. Earlier mean frames are retrospective reconstructions.',ha='center',fontsize=8)
    fig.text(.5,.015,'Posterior mean geometry; no Gaussian shape covariance is depicted as epistemic uncertainty. No native Isaac claim.',ha='center',fontsize=8)
    writer=imageio_ffmpeg.write_frames(str(output),(1536,576),fps=fps,codec='libx264',quality=7,
        macro_block_size=2,ffmpeg_log_level='error',output_params=['-threads','2','-movflags','+faststart']);writer.send(None)
    clipped=[0,0,0]
    try:
        for frame,t in enumerate(times):
            title.set_text(caption+f' | t={t:.2f}s'+(' | FORECAST EVALUATION' if t>1.5 else ''))
            for j,((ax,mesh),q) in enumerate(zip(panels,trajectories)):
                mesh.set_verts(q[frame,faces]);ax.view_init(elev=22+3*np.sin(np.pi*t/3),azim=-65+30*t/3)
                clipped[j]=max(clipped[j],int(np.any((q[frame]<limits[:,0])|(q[frame]>limits[:,1]),axis=1).sum()))
            fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            if frame==0:plt.imsave(output.with_suffix('.first-frame.png'),rgb)
            writer.send(rgb)
    finally:writer.close();plt.close(fig)
    reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24');info=next(reader);count=sum(1 for _ in reader)
    if count!=len(times) or tuple(info['size'])!=(1536,576) or abs(info['duration']-len(times)/fps)>1/fps:raise ValueError('Encoded frame/size/duration mismatch')
    return dict(output=str(output),sha256=sha(output),bytes=output.stat().st_size,frames=count,fps=fps,duration_s=info['duration'],
        limits_m=limits.tolist(),maximum_vertices_outside_fixed_bounds=clipped,motion_magnification=1.,same_camera_scale_colors=True)


def run(evaluation_path,evaluation_sha,output,*,include_worst=False):
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    evaluation,execution,frozen,by_index=gate(evaluation_path,evaluation_sha)
    selected=[(16,'PRESELECTED_RESERVED'),(17,'PRESELECTED_RESERVED')]
    if include_worst:
        worst=max(range(16,48),key=lambda i:(by_index[i]['states']['no_additional_view']['A_after1p5s_MSE_m2'],-i))
        if worst not in [16,17]:selected.append((worst,'POSTHOC_WORST_NO_VIEW_FUTURE_ERROR_FAILURE_ILLUSTRATION'))
    destination=resolve(output)
    if destination.exists():raise ValueError('Fresh LOCAL output directory required')
    destination.mkdir(parents=True)
    plan=read(checked(execution['cohort_plan']));public=read(checked(plan['public_hardware']))
    rest=np.asarray(public['rest_points_m'],dtype=np.float64);faces=np.asarray(public['surface_faces'],dtype=int)
    rows=[]
    for index,selection in selected:
        value=by_index[index];state_meta=frozen[index][1]
        with np.load(checked(state_meta['states']),allow_pickle=False) as raw:
            if not np.array_equal(raw['rest_points_m'],rest) or not np.array_equal(raw['times_s'],np.arange(61)/20):raise ValueError('Saved state rest/clock differs')
            active=value['selected_active_view'];times=raw['times_s'].copy()
            means=[rest+raw[label+'__A__mean_displacement_m'] for label in ['no_additional_view',f'fixed_view_{active}']]
        with np.load(checked(value['capture_pairs']['A']['trajectory']),allow_pickle=False) as raw:
            truth=raw['positions_m'].copy()
            if not np.array_equal(raw['times_s'],times):raise ValueError('Finite/state clocks differ')
        row=render_sequence(rest,faces,times,[truth,*means],destination/f'object-{index:03d}.mp4',caption=f'Object{index} | '+selection)
        row.update(object_index=index,selection=selection,active_view_index=active,
            future_A_after1p5s_coordinate_RMSE_m={name:float(np.sqrt(np.mean((q[31:]-truth[31:])**2))) for name,q in zip(['no_extra_view','active_view'],means)},
            predictive_states_sha256=state_meta['states']['sha256'],finite_trajectory_sha256=value['capture_pairs']['A']['trajectory']['sha256'])
        rows.append(row)
    metadata=dict(status='LOCAL_RESERVED_COMPARISON_VIDEOS_COMPLETED',renderer_sha256=sha(__file__),
        evaluation=dict(path=evaluation_path,sha256=evaluation_sha),preselected_object_indices=[16,17],optional_worst_selection='Largest sealed no-extra-view A31..60 MSE; explicitly posthoc failure illustration.',
        videos=rows,raw_videos_LOCAL=True,CPU_threads=2,GPU_used=False,new_inference_or_physics=False,calibrated_coverage_depicted=False,
        duration_note='61inclusive timestamps0..3s occupy3.05s at20fps.',labels='Independent FEM actual, posterior means conditional on sparse depths; earlier frames are retrospective, forecast evaluationt>1.5s.')
    (destination/'render-manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n');return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation',required=True);parser.add_argument('--evaluation-sha256',required=True);parser.add_argument('--output',required=True);parser.add_argument('--include-worst',action='store_true')
    args=parser.parse_args();result=run(args.evaluation,args.evaluation_sha256,args.output,include_worst=args.include_worst)
    print(json.dumps({'videos':len(result['videos']),'raw_videos_LOCAL':True}))
