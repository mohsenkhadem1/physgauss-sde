"""Posthoc worst/median object static-equilibrium orbit videos; no refits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,itertools,json,subprocess,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
import imageio_ffmpeg
from scripts.compositional_neural_mechanics import unknown_action_campaign as primary
from scripts.compositional_neural_mechanics_reporting.visual_bridge_development import solve
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.visual_metric_bridge import state_surface,Camera,render_scene
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=primary.ROOT;R=primary.R;OUT=ROOT/'runs/compositional-neural-mechanics-v16/unknown-result-videos';PLAN=R/'unknown-result-video-plan.json';VIDEO=R/'videos'

def prepare():
 if PLAN.exists()or OUT.exists():raise ValueError('Preserve prior video plan/output')
 auditpath=R/'unknown-action-final-independent-audit.json';audit=json.loads(auditpath.read_text())
 if not audit.get('passed')or audit.get('errors'):raise ValueError('Passed final audit required')
 for p,h in audit['evidence_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('Primary evidence changed')
 rows=[r for r in audit['rows']if r['model']=='shared_laws_fitted'and r['policy']=='target_task_variance']
 ranking=sorted([(float(np.mean([r['position_coordinate_rmse_m']for r in rows if r['world']==i])),i)for i in range(6)])
 cases=[]
 for label,(_,i)in [('worst_after',ranking[-1]),('median_rank',ranking[len(ranking)//2])]:
  command=max([r for r in rows if r['world']==i],key=lambda r:(r['position_coordinate_rmse_m'],-r['command']))['command'];cases.append(dict(label=label,world=i,command=command,selection='Explicit posthoc ranking bymean afterprobe heldpositionerror; withinobject worstcommand. No modelselection.'))
 plan=dict(scope=__doc__,cases=cases,object_ranking=ranking,source_sha256=closure([Path(__file__).resolve()]),input_sha256={**audit['evidence_sha256'],str(auditpath.relative_to(ROOT)):sha(auditpath)},runtime=dict(python=str(Path(sys.executable).absolute()),torch=torch.__version__),tracked_tolerance_m=1e-9,dense_grid=7,surface_cells=[6,6,6],frames=48,fps=12,width=256,height=192,workers=2,appearance='MaterialqRGB=.2+.6q; fixedgeometryappearance nohiddenlabels',labels='Static equilibrium; camera orbit only; no dynamicforecast/interpolation. Dense343point visualmetric distinctfromprimary26trackedpointscore.')
 write(PLAN,plan)

def guard():
 p=json.loads(PLAN.read_text())
 if p['runtime']!=dict(python=str(Path(sys.executable).absolute()),torch=torch.__version__):raise ValueError('Use original phystwin-env runtime for replay')
 for table in ('source_sha256','input_sha256'):
  for path,h in p[table].items():
   if sha(ROOT/path)!=h:raise ValueError('Video source/input changed')
 return p

def clip(case):
 torch.set_num_threads(1);p=guard();i=case['world'];j=case['command'];folder=primary.B/f'world{i}';target=OUT/case['label'];target.mkdir(parents=True,exist_ok=False);public=json.loads((folder/'public.json').read_text());g=GeometrySpec(**public['geometry']);pose=np.asarray(public['held_poses'][j]);modeldir=folder/'shared_laws_fitted';selection=json.loads((modeldir/'selection.json').read_text())['policies']['target_task_variance'];initial=json.loads((modeldir/'initial-fit.json').read_text())['parameters'];after=json.loads((modeldir/'fits.json').read_text())[selection]['parameters'];scene=json.loads((folder/'private/scene.json').read_text());dummy=[dict(pose=np.zeros((2,6)),points=POINTS,displacement=np.zeros_like(POINTS))];dense=np.array(list(itertools.product(np.linspace(0,1,7),repeat=3)));meshes={};fields={};tracked={};checks={};physics={}
 with np.load(modeldir/'forecasts.npz')as predictions,np.load(folder/'private/held-truth.npz')as truth:
  expected={'actual':truth[f'held{j}'],'before':predictions['initial_mean'][j*78:(j+1)*78].reshape(26,3),'after':predictions[selection+'_mean'][j*78:(j+1)*78].reshape(26,3)}
 for name,theta,learned,n in [('actual',pack(scene),False,4),('before',initial,True,2),('after',after,True,2)]:
  estimator=build(g,public['categories'],dummy,learned);estimator.kw['n']=n;model,state,metrics=solve(estimator,theta,pose);u=state['displacement_local_m'];tracked[name]=metrics['displacement'];error=float(np.max(abs(tracked[name]-expected[name])));checks[name]=error
  if error>1e-9:raise ValueError(f'Primary replay mismatch {name}:{error}')
  fields[name]=model.sample_material(u,dense);meshes[name]=state_surface(model,u,cells=(6,6,6));physics[name]=dict(wrenches=metrics['wrenches'].tolist(),residual_N=metrics['residual_N'],minimum_J=metrics['minimum_J'])
  npz(target/(name+'-state.npz'),rest=model.rest,displacement=u,elements=model.elements,dense_material=dense,dense_displacement=fields[name],tracked_displacement=tracked[name],wrenches=metrics['wrenches'])
 dense_error={name:float(np.sqrt(np.mean((fields[name]-fields['actual'])**2)))for name in ('before','after')};primary_error={name:float(np.sqrt(np.mean((tracked[name]-tracked['actual'])**2)))for name in ('before','after')}
 # No render occurs until all three tracked-state reproductions pass.
 center=g.map(np.array([[.5,.5,.5]]))[0];radius=max(g.dimensions)*2.4;frames=target/'frames';frames.mkdir();preview=Image.new('RGB',(768,260*3),'white')
 for k in range(48):
  angle=-.6+1.2*k/47;offset=np.array([np.sin(angle),-np.cos(angle),.35]);eye=center+radius*offset/np.linalg.norm(offset);cam=Camera(tuple(eye),tuple(center),256,192,250.)
  image=Image.new('RGB',(768,260),'white');draw=ImageDraw.Draw(image)
  for column,name in enumerate(('actual','before','after')):
   rgb=render_scene([meshes[name]],cam)['rgb'];image.paste(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)),(column*256,68));title={'actual':'Actual analytic n4','before':'Inferred before probe n2','after':'Inferred after '+selection+' n2'}[name];draw.text((column*256+4,4),title,fill='black')
   draw.text((column*256+4,20),'Static equilibrium / camera orbit',fill='black')
   if name!='actual':draw.text((column*256+4,36),f'26pt RMSE {primary_error[name]*1000:.3f} mm',fill='black');draw.text((column*256+4,52),f'Dense343 RMSE {dense_error[name]*1000:.3f} mm',fill='black')
   else:draw.text((4,36),f'Object {i}, held command {j}',fill='black');draw.text((4,52),'No motion interpolation',fill='black')
  image.save(frames/f'{k:03d}.png')
  if k in (0,24,47):preview.paste(image,(0,[0,24,47].index(k)*260))
 previewpath=target/'preview.png';preview.save(previewpath);VIDEO.mkdir(exist_ok=True);video=VIDEO/f'unknown-{case["label"]}-world{i}.mp4'
 if video.exists():raise ValueError('Preserve existingclip')
 ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-framerate','12','-i',str(frames/'%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
 subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(video),'-f','null','-'],check=True)
 reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);count=sum(1 for _ in reader)
 if count!=48 or abs(meta['fps']-12)>1e-8 or abs(meta['duration']-4)>.02 or tuple(meta['size'])!=(768,260):raise ValueError('Decoded clip timing/dimensions mismatch')
 result=dict(case=case,plan_sha256=sha(PLAN),tracked_max_abs_replay_error_m=checks,primary26_rmse_m=primary_error,dense343_rmse_m=dense_error,selected_action=selection,physical_states=physics,video=str(video.relative_to(ROOT)),video_sha256=sha(video),bytes=video.stat().st_size,decoded_frames=count,fps=meta['fps'],duration_seconds=meta['duration'],dimensions=meta['size'],preview=str(previewpath.relative_to(ROOT)),preview_sha256=sha(previewpath),visual_review='Pending human/model image inspection',scope=p['labels']);write(target/'complete.json',result);return result

def run():
 p=guard();OUT.mkdir(exist_ok=False)
 with ProcessPoolExecutor(max_workers=2)as pool:results=list(pool.map(clip,p['cases']))
 guard();write(R/'unknown-result-video-metadata.json',dict(plan_sha256=sha(PLAN),clips=results))
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=parser.parse_args();prepare()if a.prepare else run()
