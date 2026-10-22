"""Recorded Newton GPU versus independent CPU discrete reference, true scale.

No GPU libraries or simulations. Shared camera, held recorded states, no motion
interpolation, fitted alignment or geometric amplification. Completed failed
comparisons require explicit --allow-failed and a conspicuous failure title.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.video_block_gravity_suite import draw_panel,volume_weights,camera

ROOT=Path(__file__).resolve().parents[1]
EXTERNAL_DEFAULT='runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolve(path,root=ROOT):
    p=(Path(root)/path).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('Repository-contained source/capture path required')
    return p
def record(path,root=ROOT):return dict(path=str(Path(path).resolve().relative_to(Path(root).resolve())),sha256=sha(path))
def checked_record(r,root=ROOT):
    p=resolve(r['path'],root)
    if sha(p)!=r['sha256']:raise ValueError('Changed hash-bound source or capture: '+r['path'])
    return p
def source_map(values,root):
    if not values:raise ValueError('Nonempty physical source hash map required')
    for name,digest in values.items():checked_record(dict(path=name,sha256=digest),root)

def schedule(repeats=4,fps=30,final_hold=15):
    if (repeats,fps,final_hold)!=(4,30,15):raise ValueError('Fixed recorded240Hz/4holds/30fps/15finalholds schedule required')
    return np.r_[np.repeat(np.arange(25),repeats),np.full(final_hold,24)],240*repeats/fps

def load_pair(*,gpu_case,gpu_capture,gpu_summary,gpu_protocol,gpu_protocol_sha256,
              cpu_case,cpu_capture,cpu_summary,cpu_protocol,cpu_protocol_sha256,
              allow_failed=False,root=ROOT,external_root=EXTERNAL_DEFAULT):
    """Explicit case paths; verify completion and source/capture hashes before arrays."""
    root=Path(root).resolve()
    paths={name:resolve(value,root) for name,value in locals().copy().items() if name in ('gpu_case','gpu_capture','gpu_summary','gpu_protocol','cpu_case','cpu_capture','cpu_summary','cpu_protocol')}
    gm=json.loads(paths['gpu_summary'].read_text());cm=json.loads(paths['cpu_summary'].read_text())
    if gm.get('completed') is not True or cm.get('completed') is not True:raise ValueError('Both full CPU/GPU executions must be completed before rendering')
    for prefix,digest,summary in [('gpu',gpu_protocol_sha256,gm),('cpu',cpu_protocol_sha256,cm)]:
        if sha(paths[prefix+'_protocol'])!=digest:raise ValueError('Expected frozen protocol SHA required')
        if checked_record(summary['protocol'],root)!=paths[prefix+'_protocol']:raise ValueError('Summary protocol path differs')
    gp=json.loads(paths['gpu_protocol'].read_text());cp=json.loads(paths['cpu_protocol'].read_text())
    source_map(gp['source_sha256'],root);source_map(cp['source_sha256'],root)
    for name,digest in gp.get('external_source_sha256',{}).items():checked_record(dict(path=name,sha256=digest),resolve(external_root,root))
    for name,digest in gp['CPU_reference_sha256'].items():checked_record(dict(path=name,sha256=digest),root)
    cpu_protocols=gp['CPU_protocol_sha256']
    if isinstance(cpu_protocols,dict):
        if cpu_protocols.get(str(paths['cpu_protocol'].relative_to(root)))!=cpu_protocol_sha256:raise ValueError('CPU protocol not pinned by GPU protocol')
        for name,digest in cpu_protocols.items():checked_record(dict(path=name,sha256=digest),root)
    elif cpu_protocols!=cpu_protocol_sha256:raise ValueError('CPU protocol differs from declared physical reference')
    g=json.loads(paths['gpu_case'].read_text());c=json.loads(paths['cpu_case'].read_text())
    if sum(row==g for row in gm.get('cases',[]))!=1 or sum(row==c for row in cm.get('cases',[]))!=1:raise ValueError('Per-case result differs from completed summary')
    method,hz,iterations=g['method'],g['hz'],g['iterations']
    if (c['method'],c['hz'])!=(method,hz) or dict(method=method,hz=hz,iterations=iterations) not in gp['jobs']:raise ValueError('Undeclared or mismatched selected case')
    if g.get('completed') is not True or c.get('completed') is not True:raise ValueError('Partial captures cannot represent complete comparisons')
    cpu_ok=c['maximum_physical_original_free_R_N']<=1e-9 and c['minimum_J']>.1 and c['maximum_constraint_defect_m']==0
    checked=bool(g.get('passed') is True and cpu_ok)
    if not checked and not allow_failed:raise ValueError('Failed comparison requires explicit allow_failed/failure title')
    for prefix,case in [('gpu',g),('cpu',c)]:
        if checked_record(case['capture'],root)!=paths[prefix+'_capture']:raise ValueError('Explicit capture differs from case record')
    for name in ('cpu_capture','cpu_case'):
        if gp['CPU_reference_sha256'].get(str(paths[name].relative_to(root)))!=sha(paths[name]):raise ValueError('Selected CPU capture/result not pinned by GPU protocol')
    with np.load(paths['gpu_capture'],allow_pickle=False) as z:gpu={k:z[k].copy() for k in z.files}
    with np.load(paths['cpu_capture'],allow_pickle=False) as z:cpu={k:z[k].copy() for k in z.files}
    for key in ('rest_m','tets','faces','material_probes_m'):np.testing.assert_array_equal(gpu[key],cpu[key])
    for key in ('times_s','query_times_s'):np.testing.assert_allclose(gpu[key],cpu[key],rtol=0,atol=1e-12)
    for key in ('physical_masses_kg','applied_forces_N'):np.testing.assert_allclose(gpu[key],cpu[key],rtol=0,atol=1e-14)
    mask=cpu['constraint_mask']
    if mask.dtype!=np.bool_ or mask.shape!=cpu['rest_m'].shape or not np.array_equal(mask.all(1),mask.any(1)):raise ValueError('Complete fixed-node constraints required')
    np.testing.assert_array_equal(gpu['fixed_node_ids'],np.flatnonzero(mask.all(1)))
    steps=round(.1*hz);ids=np.arange(25)*(hz//240)
    if hz%240 or ids[-1]!=steps:raise ValueError('Exact common240Hz recorded frame indices required')
    for data in (gpu,cpu):
        if data['positions_m'].shape!=(steps+1,len(data['rest_m']),3) or not np.isfinite(data['positions_m']).all():raise ValueError('Full finite recorded nodal history required')
        np.testing.assert_allclose(data['times_s'][ids],np.arange(25)/240,rtol=0,atol=1e-12)
        np.testing.assert_allclose(data['query_times_s'],np.arange(25)/240,rtol=0,atol=1e-12)
    full_error=float(np.linalg.norm(gpu['positions_m']-cpu['positions_m'],axis=-1).max())
    query_error=float(np.linalg.norm(gpu['query_positions_m']-cpu['query_positions_m'],axis=-1).max())
    np.testing.assert_allclose(full_error,g['maximum_CPU_allnode_position_error_m'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(query_error,g['maximum_CPU_fixedquery_position_error_m'],rtol=1e-12,atol=1e-12)
    if g.get('passed') is True and (full_error>20e-6 or query_error>20e-6):raise ValueError('Passing title contradicted by recorded fields')
    return dict(gpu=gpu,cpu=cpu,gpu_case=g,cpu_case=c,gpu_protocol=gp,cpu_protocol=cp,checked=checked,
        sample_ids=ids,times_s=np.arange(25)/240,gpu_capture=record(paths['gpu_capture'],root),cpu_capture=record(paths['cpu_capture'],root),
        gpu_summary=record(paths['gpu_summary'],root),cpu_summary=record(paths['cpu_summary'],root),gpu_protocol_record=record(paths['gpu_protocol'],root),cpu_protocol_record=record(paths['cpu_protocol'],root),
        gpu_case_record=record(paths['gpu_case'],root),cpu_case_record=record(paths['cpu_case'],root))

def render(output,*,root=ROOT,**inputs):
    pair=load_pair(root=root,**inputs)
    method,hz,iterations=(pair['gpu_case'][name] for name in ('method','hz','iterations'))
    out=resolve(output,root)
    if out.exists():raise FileExistsError('Fresh local media directory required')
    out.mkdir(parents=True);gpu=pair['gpu'];cpu=pair['cpu'];ids=pair['sample_ids'];gq=gpu['positions_m'][ids];cq=cpu['positions_m'][ids]
    rest=gpu['rest_m'];faces=gpu['faces'];fixed=gpu['fixed_node_ids'].astype(int);weights=volume_weights(rest,gpu['tets'])
    node_error=np.linalg.norm(gq-cq,axis=-1).max(axis=1);query_error=np.linalg.norm(gpu['query_positions_m']-cpu['query_positions_m'],axis=-1).max(axis=1)
    sequence,slow=schedule();size=(1280,640);boxes=((10,95,630,540),(650,95,1270,540))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    video=out/'newton-and-cpu-reference.mp4';writer=imageio_ffmpeg.write_frames(str(video),size,fps=30,codec='libx264',macro_block_size=2,output_params=['-crf','19','-threads','2','-movflags','+faststart']);writer.send(None)
    checked=pair['checked'];clipped=False
    try:
        for frame,i in enumerate(sequence):
            phase=frame/(len(sequence)-1);target=weights@((gq[i]+cq[i])/2)
            canvas=Image.new('RGB',size,'#f6f8fa');draw=ImageDraw.Draw(canvas)
            status='Matching-step GPU/CPU position check passed' if checked else 'FAILED comparison — accuracy check did not pass'
            draw.text((18,10),f'{method} | {hz} Hz | {iterations} VBD iterations | {status}',font=font,fill='#142b3b' if checked else '#b21e2b')
            draw.text((18,42),'Newton GPU simulation',font=font,fill='#142b3b');draw.text((660,42),'independent CPU reference',font=font,fill='#142b3b')
            draw.text((18,70),'Same discrete material, force and boundary | shared camera | geometry at 1x scale',font=small,fill='#344c5a')
            for q,box in ((gq[i],boxes[0]),(cq[i],boxes[1])):
                draw_panel(canvas,q,rest,faces,fixed,target,phase,box)
                width,height=box[2]-box[0],box[3]-box[1]
                for points in (q,rest):
                    uv,_,_=camera(points,target,phase,np.array([width/2,height/2+10]),width/.39)
                    clipped=clipped or bool((uv[:,0]<0).any() or (uv[:,0]>=width).any() or (uv[:,1]<0).any() or (uv[:,1]>=height).any())
            draw.text((18,556),f't = {pair["times_s"][i]:.4f} s | max nodal difference = {node_error[i]*1e6:.2f} um | fixed-query difference = {query_error[i]*1e6:.2f} um',font=font,fill='#142b3b')
            draw.text((18,588),f'{slow:g}x slow playback; recorded states held, no interpolation | 45-degree orbit | orange: fixed nodes',font=small,fill='#344c5a')
            draw.text((18,614),'Discrete-model comparison; temporal/spatial qualification separate. Neither panel is physical ground truth or a learned prediction.',font=small,fill='#344c5a')
            writer.send(np.asarray(canvas).tobytes())
            if frame==0:canvas.save(out/'first-frame.png')
            if frame==len(sequence)-1:canvas.save(out/'final-frame.png')
    finally:writer.close()
    reader=imageio_ffmpeg.read_frames(str(video));next(reader);decoded=sum(1 for _ in reader)
    if decoded!=len(sequence) or clipped:raise ValueError('Decoded count or un-clipped true-scale camera verification failed')
    info=dict(schema='recorded-Newton-vs-independent-CPU-video-v01',completed=True,checked_matching_step_case=checked,failed_case_explicitly_titled=not checked,
        method=method,hz=hz,iterations=iterations,panels=['Newton GPU simulation','independent CPU reference'],fps=30,decoded_frames=decoded,frames=len(sequence),
        playback_slow_factor=slow,final_hold_frames=15,source_query_hz=240,saved_frame_indices=ids.tolist(),times_s=pair['times_s'].tolist(),
        geometric_scale=1,camera_orbit_degrees=45,shared_camera_target='common midpoint of nominal mass centers',camera_clipped=False,
        interpolated_states=False,rigid_alignment=False,is_learned_prediction=False,is_physical_ground_truth=False,
        maximum_saved_node_error_m=float(node_error.max()),maximum_fixedquery_error_m=float(query_error.max()),
        references={k:pair[k] for k in ('gpu_capture','cpu_capture','gpu_summary','cpu_summary','gpu_protocol_record','cpu_protocol_record','gpu_case_record','cpu_case_record')},
        renderer=record(Path(__file__),root),camera_primitives=record(Path(__file__).with_name('video_block_gravity_suite.py'),root),
        raw_media_local=True,mp4_sha256=sha(video),nonclaims=pair['gpu_protocol']['nonclaims'])
    (out/'video.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n');return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for prefix in ('gpu','cpu'):
        for field in ('case','capture','summary','protocol','protocol-sha256'):p.add_argument('--'+prefix+'-'+field,required=True)
    p.add_argument('--output',required=True);p.add_argument('--allow-failed',action='store_true');p.add_argument('--external-root',default=EXTERNAL_DEFAULT)
    values=vars(p.parse_args());output=values.pop('output');result=render(output,**values)
    print(json.dumps({k:result[k] for k in ('completed','checked_matching_step_case','frames','maximum_saved_node_error_m','raw_media_local')}))
