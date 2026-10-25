"""LOCAL saved static NH80 reference versus same-parameter LINEAR ORACLE video.

No PDE/GPU/RNG/model inference. Frames change only camera pose; both presets
remain fixed saved equilibria. Complete audited negative controls are allowed,
with a prominent failed-preset-gates label. Rendering needs explicit release.
"""
import argparse,ast,json,os,resource,shutil,time
from pathlib import Path
import numpy as np
from experiments import audit_fixture_balanced_held_nh_refinement_v01 as audit
ROOT=audit.stage.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-balanced-NH-linear-video-v01'
SOURCE='experiments/render_fixture_balanced_nh_comparison_v01.py';TEST='tests/test_render_fixture_balanced_nh_comparison_v01.py'
STYLE=dict(fps=12,frames_per_load=48,frames=96,width=1280,height=720,dpi=100,displacement_scale=20.,saved_mesh_n=12,display_mesh_n=6,query_points=16,azimuth_start=-65.,azimuth_sweep=120.,elevation_degrees=25.,codec='libx264',crf=18,video_duration_s=8.,CPU=19,RSS_bytes=4*1024**3,wall_s=300,maximum_writes_bytes=128*1024**2)
sha,record,read,write=audit.stage.sha,audit.stage.record,lambda p:json.loads(Path(p).read_text()),audit.stage.write

def sources():
    todo=[SOURCE,TEST,audit.SOURCE,audit.TEST];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for module in modules:
                p=module.replace('.','/')+'.py'
                if(ROOT/p).is_file()and p not in seen:todo.append(p)
    return {p:sha(ROOT/p)for p in sorted(seen)}
def origin_gate():
    p,e,r,c=audit.metadata_barrier();base=audit.stage.LOCAL;review=read(base/'independent-review.json')
    audit.need(review.get('audit_passed')is True and review.get('source_result')==record(base/'result-LOCAL.json')and review.get('root_closure')==record(base/'root-closure.json')and review.get('source_SHA256')==p['source_SHA256'],'Completed frozen independent audit BEFORE any field read')
    audit.need(review['audit_source']==record(ROOT/audit.SOURCE)and review['audit_test']==record(ROOT/audit.TEST)and review['compact_summary']==record(base/'compact-summary.json'),'Independent procedure and compact-summary bytes')
    selected=[next(row for row in r['rows']if row['n']==12 and row['label']==label)for label in audit.stage.PILOT_LOADS]
    audit.need(len(selected)==2,'Both declared presets in declared order')
    return dict(protocol=record(base/'protocol.json'),release=record(base/'release.json'),execution=record(base/'execution.json'),result=record(base/'result-LOCAL.json'),root_closure=record(base/'root-closure.json'),review=record(base/'independent-review.json'),compact=record(base/'compact-summary.json'),captures=[row['capture']for row in selected],labels=list(audit.stage.PILOT_LOADS),pilot_qualified=bool(r['pilot_qualified']))
def prepare():
    origin=origin_gate()
    if(LOCAL/'recipe.json').exists():raise FileExistsError('New render namespace; no recipe overwrite')
    import imageio_ffmpeg
    ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe());recipe=dict(schema='fixture-balanced-NH-linear-static-video-v01',source_SHA256=sources(),origin=origin,style=STYLE,ffmpeg=dict(path=str(ffmpeg),sha256=sha(ffmpeg)),same_TRUE_material_and_fixture=True,learned_prediction=False,static_equilibria=True,camera_only_motion=True,no_displacement_ramp=True,coarsened_surface_display_only=True)
    write(LOCAL/'recipe.json',recipe);return record(LOCAL/'recipe.json')
def validate(digest):
    p=read(LOCAL/'recipe.json');r=read(LOCAL/'release.json')
    audit.need(sha(LOCAL/'recipe.json')==digest and p['source_SHA256']==sources()and p['style']==STYLE,'Exact renderer source/recipe/style')
    audit.need(r.get('approved')is True and r.get('recipe_SHA256')==digest and r.get('source_SHA256')==p['source_SHA256'],'Explicit root render release BEFORE arrays')
    audit.need(p['origin']==origin_gate(),'Unchanged terminal captures/audit/closure');audit.need(sha(p['ffmpeg']['path'])==p['ffmpeg']['sha256'],'Pinned bundled ffmpeg executable');return p

def surface_triangles(faces):
    f=np.asarray(faces)
    if f.ndim!=2 or f.shape[1]!=6 or f.dtype.kind not in'iu':raise ValueError('T6 face ordering vertices0/1/2,mid01/mid02/mid12')
    return np.concatenate([f[:,ids]for ids in((0,3,4),(3,1,5),(4,5,2),(3,5,4))])
def status_label(pilot_qualified):return 'PRESET GATES PASSED — six controls only'if pilot_qualified else 'FAILED PRESET GATES — illustrative saved state'
def guard(start):
    if set(os.sched_getaffinity(0))!={STYLE['CPU']}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Renderer CPU19/no visible CUDA')
    if any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('One thread renderer')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=STYLE['RSS_bytes']or time.monotonic()-start>STYLE['wall_s']:raise RuntimeError('Bounded300s/4GiB LOCAL rendering')
    if sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>STYLE['maximum_writes_bytes']:raise MemoryError('Owned128MiB media cap')

def render(digest):
    p=validate(digest);start=time.monotonic();guard(start)
    if(LOCAL/'comparison.mp4').exists()or(LOCAL/'render-result.json').exists():raise FileExistsError('No implicit render retry/overwrite')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio_ffmpeg
    from PIL import Image
    # The complete audit/hash/release barriers above precede FIRST NPZ access.
    X,_,faces,fixed,_=audit.stage.resolution.geometry(6);tri=surface_triangles(faces);fixed_faces=np.all(X[tri][:,:,0]==0,axis=1)|np.all(X[tri][:,:,0]==.36,axis=1);colors=np.where(fixed_faces[:,None],np.array([[.16,.40,.87,.95]]),np.array([[.49,.30,.75,.88]]));presets=[]
    for label,capture in zip(p['origin']['labels'],p['origin']['captures']):
        with np.load(ROOT/capture['path'],allow_pickle=False)as z:
            nl=z['displacement_m'].copy();linear=z['linear_displacement_m'].copy();points=z['query_points_m'].copy();rest=z['rest_m'];audit.need(rest.shape==nl.shape==linear.shape and np.isfinite(nl).all()and np.isfinite(linear).all(),'Finite identical saved nonlinear/linear geometry')
        nl_display=audit.stage.resolution.field_queries(12,nl,X);linear_display=audit.stage.resolution.field_queries(12,linear,X);q_nl=audit.stage.resolution.field_queries(12,nl,points);q_lin=audit.stage.resolution.field_queries(12,linear,points);error=float(np.sqrt(np.mean((q_nl-q_lin)**2)))
        spec=next(s for s in audit.stage.bank.load_specs()if s['id']==label);center=np.asarray(spec['center_m']);direction=np.asarray(spec['direction']);application=[center+20*audit.stage.resolution.field_queries(12,u,center[None])[0]for u in(nl,linear)];presets.append(dict(label=label,fields=[nl_display,linear_display],query_RMS_m=error,force=spec,application=application))
        del nl,linear;guard(start)
    all_positions=np.concatenate([X]+[X+20*u for preset in presets for u in preset['fields']]);lower=all_positions.min(axis=0)-.035;upper=all_positions.max(axis=0)+.035;span=upper-lower
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f8fafc');axes=[fig.add_axes([.015,.12,.48,.71],projection='3d'),fig.add_axes([.505,.12,.48,.71],projection='3d')];fig.subplots_adjust(0,0,1,1);artists=[];titles=['NONLINEAR REFERENCE (NH80)','LINEAR ORACLE — same true parameters'];rest_edges=[]
    # Rest outline is a visual reference, not a second deformation state.
    import itertools
    corners=np.array(list(itertools.product((0.,.36),(0.,.08),(0.,.08))))
    for i,a in enumerate(corners):
        for j,b in enumerate(corners):
            if j>i and np.count_nonzero(a!=b)==1:rest_edges.append((a,b))
    fixture=audit.stage.bank.bank.layout_spec(audit.stage.PILOT_LAYOUT);c=np.asarray(fixture['center_m']);std=np.asarray(fixture['std_m']);angle=np.linspace(0,2*np.pi,80);ellipse=np.column_stack([c[0]+2*std[0]*np.cos(angle),c[1]+2*std[1]*np.sin(angle),np.zeros_like(angle)])
    header=fig.text(.5,.94,'',ha='center',fontsize=17,weight='bold',color='#172033');banner=fig.text(.5,.885,status_label(p['origin']['pilot_qualified']),ha='center',fontsize=12,weight='bold',color='#17653c'if p['origin']['pilot_qualified']else'#b00020');metric=fig.text(.5,.075,'',ha='center',fontsize=12,color='#172033');fig.text(.5,.025,'STATIC equilibria · displacement ×20 · camera motion only · n12 P2 fields, n6 surface display only',ha='center',fontsize=11,color='#334155')
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=13,pad=7,color='#172033');ax.set_box_aspect(span);ax.set_xlim(lower[0],upper[0]);ax.set_ylim(lower[1],upper[1]);ax.set_zlim(lower[2],upper[2]);ax.set_axis_off()
        for a,b in rest_edges:ax.plot(*np.array([a,b]).T,color='#64748b',alpha=.5,lw=.6,linestyle='--')
        ax.plot(*ellipse.T,color='#e87516',lw=2.7);ax.text(c[0],c[1],-.027,'compliant fixture',fontsize=8,color='#b85705',ha='center')
        collection=Poly3DCollection([],edgecolors=(.15,.20,.30,.13),linewidths=.15,zsort='average');ax.add_collection3d(collection);artists.append(collection)
    path=LOCAL/'comparison.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1280,720),fps=12,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-preset','medium','-threads','1','-movflags','+faststart','-metadata','creation_time=']);writer.send(None);arrows=[]
    try:
        for frame in range(96):
            load=frame//48;local=frame%48;preset=presets[load]
            if local==0:
                for arrow in arrows:arrow.remove()
                arrows=[];header.set_text(f"Held load {load+1}/2 · {preset['label'].replace('held-','')}");metric.set_text(f"Linear-oracle mismatch at 16 public markers: {preset['query_RMS_m']*1e6:.2f} µm coordinate RMS   |   force 0.06 N   |   blue: both fixed ends")
                for j,(ax,artist)in enumerate(zip(axes,artists)):
                    artist.set_verts((X+20*preset['fields'][j])[tri]);facecolors=colors.copy()
                    if j==1:facecolors[~fixed_faces]=[.10,.60,.61,.88]
                    artist.set_facecolor(facecolors);point=preset['application'][j];d=np.asarray(preset['force']['direction']);arrow=ax.quiver(*(point-.045*d),*(.045*d),color='#d73925',arrow_length_ratio=.25,linewidth=2);arrows.append(arrow)
            for ax in axes:ax.view_init(elev=25+5*np.sin(np.pi*local/47),azim=-65+120*local/47)
            fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();writer.send(rgb.tobytes());guard(start)
    finally:writer.close();plt.close(fig)
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(reader);count=0;previews=[]
    for raw in reader:
        if count in(0,48):
            preview=LOCAL/f'preview-frame-{count:02d}.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(720,1280,3)).save(preview);previews.append(record(preview))
        count+=1
    audit.need(count==96 and tuple(meta['size'])==(1280,720)and abs(meta['fps']-12)<1e-9,'Decode all96 actual video frames at12fps')
    result=dict(completed=True,recipe=record(LOCAL/'recipe.json'),release=record(LOCAL/'release.json'),video=record(path),previews=previews,decoded_frames=count,decoded_size=list(meta['size']),fps=meta['fps'],duration_s=count/meta['fps'],labels=p['origin']['labels'],displacement_scale=20,static_equilibria=True,camera_only=True,learned_prediction=False,same_true_parameters=True,query_RMS_m=[v['query_RMS_m']for v in presets],pilot_qualified=p['origin']['pilot_qualified'],failed_gate_overlay=not p['origin']['pilot_qualified'],surface_display_coarsened_only=True,wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'render-result.json',result);return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','render']);ap.add_argument('--sha');a=ap.parse_args();print(prepare()if a.action=='prepare'else render(a.sha))
