"""Prespecified actual/previous guarded/selected-model exploratory videos."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import imageio_ffmpeg
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json,check_deadline
from scripts.contact_benchmark_v12.data import load
from scripts.contact_benchmark_v12.scene import load_scene
from scripts.external_benchmark.evaluate import expand_prediction

CASES=[('regression',dict(case='double_stretch_sloth')),
       ('untouched_confirmation',dict(case='single_lift_zebra')),
       ('untouched_confirmation',dict(subset='object-soft',task=1,episode=10)),
       ('fresh_confirmation',dict(case='single_push_sloth')),
       ('fresh_confirmation',dict(subset='spatial-soft',task=2,episode=20))]


def render(cohort,spec):
    freeze=REPORT/'model-selection-freeze.json'
    if not freeze.exists():raise RuntimeError(f'Refusing rendering before {freeze.name}')
    data=load(spec);name=data['name'];folder=BASE/'model-selection-evaluation'/cohort/name;source=folder/'predictions.npz'
    metadata_path=folder/'prediction-metadata.json'
    if not source.exists() or not metadata_path.exists():raise FileNotFoundError(source)
    archive=np.load(source);method='model_selection_selected';metadata=json.loads(metadata_path.read_text())
    previous_source=BASE/'feasibility-evaluation'/cohort/name/'predictions.npz'
    previous_key='feasible_object_frame' if data['dataset']=='PhysTwin' else 'feasible_contact_scene_state'
    previous=np.load(previous_source)[previous_key]
    selected=archive[method]
    count=len(data['evaluation']['points'][0])
    clouds=[data['evaluation']['points'],expand_prediction(data,previous)[:,:count],expand_prediction(data,selected)[:,:count]]
    T=len(data['times'])
    if any(len(c)!=T or c.shape[1]!=count for c in clouds):raise ValueError('Incompatible rendering geometry shapes')
    masks=data['evaluation']['mask'];scene=load_scene(data) if data['dataset']=='SoftVTBench' else None
    scenery=np.concatenate([m['vertices'][::max(1,len(m['vertices'])//1800)] for m in scene.meshes]) if scene and scene.meshes else np.empty((0,3))
    points=np.concatenate([c.reshape(-1,3) for c in clouds]+[scenery]);points=points[np.isfinite(points).all(1)]
    center=(points.min(0)+points.max(0))/2;radius=max(np.linalg.norm(points-center,axis=1).max(),.001);scale=215/radius
    shape_radius=0.
    for c in clouds:
        for t in range(T):
            valid=masks[t]&np.isfinite(c[t]).all(1)
            if valid.any():shape_radius=max(shape_radius,float(np.linalg.norm(c[t,valid]-c[t,valid].mean(0),axis=1).max()))
    shape_scale=135/max(shape_radius,.001)
    initial=clouds[0][0];rgb=np.c_[(initial[:,0]-initial[:,0].min())/(np.ptp(initial[:,0])+1e-9),(initial[:,1]-initial[:,1].min())/(np.ptp(initial[:,1])+1e-9),np.full(count,.5)]
    colors=(60+150*rgb).astype(np.uint8)
    fps=1/float(data['times'][1]-data['times'][0]);video=REPORT/f'model-selection-{cohort}-{name}.mp4';temp=video.with_suffix('.raw.mp4')
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1536,960))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    snapshots=set(np.linspace(0,T-1,4,dtype=int).tolist());snapshots.add(data['train_end'])
    configuration=metadata['selected_configuration'];label=f"{configuration['family']} / {configuration['method']} / {configuration['strength']:g}"
    titles=['Recorded geometry','Previous guarded predictor','Selected model: '+configuration['family']]
    def draw(img,p,valid,basis,depth,origin,pixelscale,offset,ycenter,top,bottom,radius_px=2,grey=False):
        ids=np.flatnonzero(valid)
        if not len(ids):return
        order=ids[np.argsort((p[ids]-origin)@depth)]
        uv=(p[order]-origin)@basis*pixelscale;uv=np.c_[uv[:,0]+256,ycenter-uv[:,1]]
        for i,(x,y) in zip(order,uv):
            if 5<x<507 and top<y<bottom:cv2.circle(img,(int(x)+offset,int(y)),radius_px,(185,185,185) if grey else tuple(int(a) for a in colors[i]),-1,cv2.LINE_AA)
    for t in range(T):
        check_deadline()
        img=np.full((960,1536,3),248,np.uint8);angle=.35+.25*np.sin(2*np.pi*t/max(T-1,1));up=.3
        right=np.array([np.cos(angle),np.sin(angle),0]);vertical=np.array([-np.sin(angle)*np.sin(up),np.cos(angle)*np.sin(up),np.cos(up)])
        basis=np.c_[right,vertical];depth=np.cross(right,vertical)
        for panel,c in enumerate(clouds):
            offset=512*panel;valid=masks[t]&np.isfinite(c[t]).all(1)
            cv2.putText(img,titles[panel],(offset+12,32),0,.52,(25,25,25),1)
            cv2.putText(img,'World: shared scale' if panel<2 else label,(offset+12,54),0,.43,(80,80,80),1)
            draw(img,scenery,np.ones(len(scenery),bool),basis,depth,center,scale,offset,315,65,575,1,True)
            draw(img,c[t],valid,basis,depth,center,scale,offset,315,65,575)
            if not valid.any():cv2.putText(img,'NO FINITE PREDICTION',(offset+20,315),0,.55,(30,30,180),2)
            bar=.05 if radius>.1 else .02;px=int(scale*bar);cv2.line(img,(offset+20,575),(offset+20+px,575),(40,40,40),2)
            cv2.putText(img,f'{bar*1000:g} mm',(offset+20,563),0,.42,(40,40,40),1)
            cv2.putText(img,'Shape close-up: independently centered',(offset+12,612),0,.47,(40,40,40),1)
            if valid.any():draw(img,c[t],valid,basis,depth,c[t,valid].mean(0),shape_scale,offset,757,630,900,3)
            bar=.02;px=int(shape_scale*bar);cv2.line(img,(offset+20,900),(offset+20+px,900),(40,40,40),2)
            cv2.putText(img,'20 mm',(offset+20,888),0,.42,(40,40,40),1)
        phase='OBSERVED PREFIX' if t<data['train_end'] else 'HELD-OUT PREDICTION'
        cv2.putText(img,f'{cohort} | {name} | frame {t}/{T-1} | {phase}',(15,931),0,.52,(25,25,25),1)
        cv2.putText(img,'Geometry playback, not raw RGB. Shared scale within rows; no deformation exaggeration. Grey: initial static meshes.',(15,952),0,.45,(60,60,60),1)
        writer.write(img)
        if t in snapshots:cv2.imwrite(str(video.with_name(video.stem+f'-frame{t}.png')),img)
    writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(temp),'-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True);temp.unlink()
    write_json(video.with_suffix('.json'),dict(cohort=cohort,case=name,selection='Prespecified in model-selection-video-selection.json before exploratory scores',primary_method=method,selected_configuration=configuration,switched_from_previous=metadata['switched_from_previous'],frames=T,fps=fps,fit_end=data['train_end'],source_prediction=str(source.relative_to(ROOT)),prediction_sha256=sha256(source),previous_prediction=str(previous_source.relative_to(ROOT)),previous_prediction_sha256=sha256(previous_source),metadata_sha256=sha256(metadata_path),raw_input_sha256=sha256(data['source']),freeze_sha256=sha256(freeze),renderer_sha256=sha256(Path(__file__)),scene=scene.metadata() if scene else None,world_scale_px_per_m=scale,shape_scale_px_per_m=shape_scale,snapshot_frames=sorted(snapshots),scope='Exploratory model selection, not independent confirmation; geometry playback uses evaluator only for display/framing; original frame rate, shared scale, no exaggeration'))
    print(video,flush=True)
    return video


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');parser.add_argument('--case');parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if not args.all and not args.case:parser.error('Choose --all or --case NAME from prespecified cases')
    selected=[]
    for cohort,spec in CASES:
        name=spec.get('case',f"{spec.get('subset')}-task{spec.get('task')}-demo{spec.get('episode')}")
        if args.all or args.case==name:selected.append((cohort,spec,name))
    if not selected:parser.error('Unknown prespecified case')
    while selected:
        check_deadline();pending=[]
        for cohort,spec,name in selected:
            out=REPORT/f'model-selection-{cohort}-{name}.json'
            folder=BASE/'model-selection-evaluation'/cohort/name
            ready=(REPORT/'model-selection-freeze.json').exists() and (folder/'metrics.json').exists()
            if out.exists():continue
            if ready:render(cohort,spec)
            elif not args.watch:raise FileNotFoundError(folder/'metrics.json')
            else:pending.append((cohort,spec,name))
        selected=pending
        if selected:time.sleep(20)
