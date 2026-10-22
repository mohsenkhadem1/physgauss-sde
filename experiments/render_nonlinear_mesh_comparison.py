"""One local unconverged2/16/128-tet mechanics clip; CPU rendering only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
try:
    import render_point_prior_comparison as graphics
    from campaign_nonlinear_excitation import command,boundary_faces
except ImportError:
    from experiments import render_point_prior_comparison as graphics
    from experiments.campaign_nonlinear_excitation import command,boundary_faces
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(name):raise ValueError('Canonical repository-relative path required')
    path=(ROOT/p).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Repository escape')
    return path

def checked(rec):
    path=resolve(rec['path'])
    if sha(path)!=rec['sha256']:raise ValueError('Frozen mesh-render artifact changed: '+rec['path'])
    return path

def record(path):return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(path,output_prefix):
    target=resolve(path)
    if target.exists():raise ValueError('Existing mesh render recipe retained')
    base='runs/gaussian-campaign-v01/nonlinear-excitation-v01/'
    expected={2:'41f7ac1d8f711f40e04c66a223c1144b13f9ee98c2b87bbf2125fa86e18d2190',16:'95a0718906584942fc6ae665483d94db3645e2f1bd8848a5296c6f36bbcddf50',128:'3ea95d8280de4202d493a1c3421b3669a5f84698a9c8d0ec67a8787ee23b34ed'}
    protocol=dict(path='runs/operator-dev/nonlinear-excitation-v01/protocol.json',sha256='2c458912e441d69db8b44c6133591038b1ea75e8ee175e8b314e2bf49986f250')
    physics=read(checked(protocol));sources={}
    for name,digest in physics['source_sha256'].items():
        # Existing physics protocol has one explicit absolute source path.
        # Resolve that exact pinned path, never strip arbitrary prefixes.
        original=Path(name);absolute=original.resolve() if original.is_absolute() else resolve(name)
        rel=absolute.relative_to(ROOT).as_posix()
        if sha(absolute)!=digest:raise ValueError('Physics source changed')
        sources[rel]=digest
    for name in ['render_nonlinear_mesh_comparison.py','render_point_prior_comparison.py']:
        rel='experiments/'+name;sources[rel]=sha(ROOT/rel)
    cases=[]
    for n,digest in expected.items():
        trajectory=dict(path=base+f'mesh{n}-balanced_axial.npz',sha256=digest)
        checked(trajectory) # hash bytes only, no archive member or response values
        cases.append(dict(tetrahedra=n,trajectory=trajectory,summary=record(resolve(base+f'mesh{n}-balanced_axial.json'))))
    recipe=dict(schema='unconverged-nonlinear-mesh-video-v0.1',physics_protocol=protocol,cases=cases,source_sha256=sources,output_prefix=output_prefix,
        settings=dict(pattern='balanced_axial',amplitude_N_per_patch=1.,young_Pa=5000.,fixture_scale=2.,poisson=.35,density_kg_m3=625.,hz=240,duration_s=.2,fps=24,slow_playback_factor=10.,
            motion_magnification=1.,world_bounds_margin_m=.025,preview_frame_index=48,pixels=[1536,576],camera='Shared orthographic public-clock orbit -70..-25deg/elevation22+5sin(pi*t/.2)',
            colors='One viridis rest-material x field, same scale on all meshes',title='MESH COMPARISON—UNCONVERGED',
            force_checks='Exact shared48x2x3 heldforcebytes and original command() midpoint waveform',clock_checks='Exactshared49timestamps arange49/240; no interpolation/repetition'),
        CPU_threads=2,GPU_used=False,known_mesh_comparison_failures_retained=True,no_learning_or_new_physics=True,raw_MP4_PNG_LOCAL=True)
    target.parent.mkdir(parents=True,exist_ok=True);write(target,recipe);return record(target)


def validate_recipe(path,digest):
    p=read(checked(dict(path=path,sha256=digest)))
    if p.get('schema')!='unconverged-nonlinear-mesh-video-v0.1' or p.get('CPU_threads')!=2 or p.get('GPU_used') is not False:raise ValueError('Exact independent CPU render recipe required')
    for name,value in p['source_sha256'].items():checked(dict(path=name,sha256=value))
    if p['source_sha256'].get(Path(__file__).relative_to(ROOT).as_posix())!=sha(__file__):raise ValueError('Own renderer source not frozen')
    checked(p['physics_protocol'])
    for row in p['cases']:checked(row['trajectory']);checked(row['summary'])
    return p


def validate_panels(panels,summaries):
    """Exact same authored inputs/clocks, allowing only mesh geometry to vary."""
    if len(panels)!=3 or len(summaries)!=3:raise ValueError('Three fixed mesh panels required')
    times=np.arange(49)/240;midpoints=(np.arange(48)+.5)/240
    expected=np.stack([command('balanced_axial',(k+.5)*(1/240),1.,.2) for k in range(48)])
    for n,p,s in zip([2,16,128],panels,summaries):
        if (s.get('completed') is not True or s.get('tetrahedra')!=n or s.get('pattern')!='balanced_axial'
            or s.get('young_Pa')!=5000. or s.get('fixture_scale')!=2. or s.get('amplitude_N_per_patch')!=1. or s.get('hz')!=240):raise ValueError('Same nominal material/support/controls required')
        q=p['positions_m'];rest=p['rest_m'];tets=p['tets']
        if (q.dtype!=np.float64 or q.shape!=(49,len(rest),3) or not np.isfinite(q).all() or tets.shape!=(n,4)
            or not np.array_equal(p['times_s'],times) or not np.array_equal(p['held_force_midpoints_s'],midpoints)):
            raise ValueError('Same actual240Hz timestamps and valid complete geometry required')
        if not np.array_equal(p['held_patch_forces_N'],expected):raise ValueError('Exact original balanced1N known controls required')
        if not np.array_equal(q[0],rest):raise ValueError('All mesh runs must start at own declared rest')
    if any(not np.array_equal(panels[0]['held_patch_forces_N'],p['held_patch_forces_N']) for p in panels[1:]):raise ValueError('Different known command bytes across panels')
    # The conforming refinement preserves the five original material vertices.
    if any(not np.array_equal(panels[0]['rest_m'],p['rest_m'][:5]) for p in panels[1:]):raise ValueError('Different original physical rest geometry')
    return times


def render(recipe_path,digest):
    p=validate_recipe(recipe_path,digest)
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    destination=resolve(p['output_prefix'])
    if destination.exists():raise ValueError('Fresh local render directory required')
    panels=[];summaries=[]
    for case in p['cases']:
        summaries.append(read(checked(case['summary']))['summary'])
        with np.load(checked(case['trajectory']),allow_pickle=False) as z:
            panels.append({key:z[key].copy() for key in ['positions_m','rest_m','tets','times_s','held_patch_forces_N','held_force_midpoints_s']})
    times=validate_panels(panels,summaries);s=p['settings'];rest=panels[0]['rest_m'];bounds=np.stack([rest.min(0)-s['world_bounds_margin_m'],rest.max(0)+s['world_bounds_margin_m']],axis=1)
    clipped=[int(np.any((x['positions_m']<bounds[:,0])|(x['positions_m']>bounds[:,1]),axis=2).sum(1).max()) for x in panels]
    destination.mkdir(parents=True);status=dict(terminal=False,completed=False,GPU_used=False);write(destination/'execution.json',status)
    if any(clipped):
        write(destination/'failure.json',dict(reason='Fixed publicrest bounds clip geometry',clipped_nodes=clipped,bounds_m=bounds.tolist()));status.update(terminal=True);write(destination/'execution.json',status);raise ValueError('Fixed world bounds clipping; no autozoom')
    plt=graphics.plt;fig=plt.figure(figsize=(15.36,5.76),dpi=100,facecolor='#f7f9fc');axes=[]
    for j,(panel,count) in enumerate(zip(panels,[2,16,128])):
        faces=boundary_faces(panel['tets']);r=panel['rest_m'];colors=graphics.cm.viridis((r[faces].mean(1)[:,0]-rest[:,0].min())/np.ptp(rest[:,0]))
        ax=fig.add_subplot(1,3,j+1,projection='3d');ax.set_title(f'{count} tetrahedra | {len(r)} vertices',fontsize=11)
        mesh=graphics.Poly3DCollection(r[faces],facecolors=colors,edgecolors='#34465b',linewidths=.3,alpha=.85);ax.add_collection3d(mesh)
        for axis,limits in zip('xyz',bounds):getattr(ax,'set_'+axis+'lim')(limits);getattr(ax,'set_'+axis+'label')(axis+' (m)',fontsize=8)
        ax.set_box_aspect(bounds[:,1]-bounds[:,0]);ax.set_proj_type('ortho');ax.tick_params(labelsize=7);ax.grid(False);axes.append((ax,mesh,faces))
    fig.subplots_adjust(left=.02,right=.98,bottom=.14,top=.78,wspace=.04)
    fig.text(.5,.96,s['title'],ha='center',color='#9d1b20',fontsize=17,weight='bold')
    fig.text(.5,.905,'Same 1 N balanced Gaussian patch forces · E = 5000 Pa · compliant support scale = 2',ha='center',fontsize=10)
    clock=fig.text(.5,.85,'',ha='center',fontsize=11)
    fig.text(.5,.075,'10× slower playback; TRUE SCALE, no displacement magnification. Shared world bounds, rest-material colors and orbit.',ha='center',fontsize=9)
    fig.text(.5,.025,'Finite mechanics at three meshes; mesh convergence FAILED. No learned prediction, reference truth, or continuum claim.',ha='center',fontsize=9)
    video=destination/'mesh-comparison-unconverged.mp4';preview=destination/'mesh-comparison-preview.png'
    writer=graphics.imageio_ffmpeg.write_frames(str(video),(1536,576),fps=s['fps'],codec='libx264',quality=7,macro_block_size=2,ffmpeg_log_level='error',output_params=['-threads','2','-movflags','+faststart']);writer.send(None)
    try:
        for frame,t in enumerate(times):
            clock.set_text(f'Actual simulation time {t:.4f} s / 0.2000 s | playback 0.1× speed')
            for (ax,mesh,faces),panel in zip(axes,panels):
                mesh.set_verts(panel['positions_m'][frame,faces]);ax.view_init(elev=22+5*np.sin(np.pi*t/.2),azim=-70+45*t/.2)
            fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();writer.send(rgb)
            if frame==s['preview_frame_index']:plt.imsave(preview,rgb)
    finally:writer.close();plt.close(fig)
    reader=graphics.imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');info=next(reader);count=sum(1 for _ in reader)
    if count!=49 or tuple(info['size'])!=(1536,576) or abs(info['duration']-49/24)>.05:raise ValueError('Encoded clock/frame verification failed')
    metadata=dict(schema='unconverged-nonlinear-mesh-video-completed-v0.1',completed=True,recipe=record(resolve(recipe_path)),physics_protocol=p['physics_protocol'],
        video=record(video),preview=record(preview),tetrahedra=[2,16,128],actual_times_s=times.tolist(),frames=count,fps=24,
        playback_slowdown_factor=10.,encoded_duration_s=info['duration'],simulation_duration_s=.2,
        inclusive_frame_note='49 saved endpoints at240Hz displayed at24fps;49/24s includes final endpoint display interval.',
        motion_magnification=1.,fixed_world_bounds_m=bounds.tolist(),same_controls_material_support_verified=True,
        maximum_nodes_outside_fixed_bounds=clipped,mesh_comparison_UNCONVERGED=True,no_learning_or_new_physics=True,CPU_threads=2,GPU_used=False,MP4_PNG_LOCAL=True)
    write(destination/'render-manifest.json',metadata);status.update(terminal=True,completed=True);write(destination/'execution.json',status);return record(destination/'render-manifest.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True);a.add_argument('--output-prefix',required=True)
    a=sub.add_parser('render');a.add_argument('--recipe',required=True);a.add_argument('--recipe-sha256',required=True)
    args=parser.parse_args()
    result=prepare(args.output,args.output_prefix) if args.action=='prepare' else render(args.recipe,args.recipe_sha256)
    print(json.dumps(result))
