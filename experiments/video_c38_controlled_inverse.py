"""Gated static POINT-LATTICE comparison; camera motion only, not dynamics.

Source/selection precedes target access. No actual data are read on import or
prepare. Rendering requires the complete initial, choice, postprobe AND final
evaluation seals. No future probe outcomes are extracted, even when authorized.
"""
from __future__ import annotations
import argparse,json,hashlib,os,signal,subprocess,time
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_full_evaluation as full
from experiments import run_campaign_c38_prediction_evaluation as evaluation
from experiments.campaign_c38_table_validation import POINTS
from experiments.campaign_c38_controlled_observations import POINT_IDS
io=evaluation.io;ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/video_c38_controlled_inverse.py'
SELECTION=[dict(private_configuration_index=i,repeat_index=0) for i in range(4)]
RECIPE=dict(schema='C38-static-unknown-physics-point-video-v01',selection=SELECTION,model='nonlinear_table',method='adaptive',target_force_Fz_N=1.,after_policy='target_risk',fps=30,frames_per_case=120,width=1536,height=960,world_scale_pixels_per_m=1500.,orbit_azimuth_degrees=[20.,65.],elevation_degrees=20.,motion='STATIC_FIELDS_CAMERA_MOTION_ONLY',geometry_scale=1.,surface_claim=False,interval_scope='Raw pointwise95%model-conditional equal-tail intervals on ONLY9qualifiedtop points; no sensor-noise addition or joint/populationcoverage',profile_units='z displacement micrometres; x is reference materialcoordinate millimetres',CPU_cores=[30,31],threads=2,hard_wall_s=180,no_GPU=True)


def sources():return sorted(set(io.source_names())|{SOURCE,'tests/test_video_c38_controlled_inverse.py','experiments/run_campaign_c38_full_evaluation.py','experiments/run_campaign_c38_prediction_evaluation.py','experiments/campaign_c38_prediction_metrics.py','experiments/campaign_c38_policy_metrics.py','experiments/campaign_c38_table_validation.py'})


def prepare(recipe_path,output,*,root=ROOT):
 root=Path(root).resolve();p=io.local(root,recipe_path)
 if p.exists() or io.local(root,output).exists():raise FileExistsError('Fresh recipe/output; preserve failures')
 r=dict(recipe=RECIPE,output=output,source_sha256={n:io.sha(root/n) for n in sources()},runtime=io.runtime(),status='FROZEN_SELECTION_BEFORE_TARGET_READS',actual_values_read=False)
 p.parent.mkdir(parents=True,exist_ok=True);io.write_new(p,r);return io.sha(p)


def validate_prediction_barrier(initial,choices,post,*,initial_record,choices_record):
 """Pure public metadata gate, before even opening private aggregate values."""
 if initial.get('completed') is not True or initial.get('count')!=64 or initial.get('model_executions')!=128 or initial.get('private_truth_access') is not False:raise ValueError('Complete initial64x2 seal required')
 ids={r['object_id'] for r in initial.get('objects',[])}
 if len(ids)!=64:raise ValueError('All64 initial identities')
 evaluation.verify_choices(choices,initial_record,ids);rows=full.validate_post_seal(post,initial_record,choices_record,ids)
 return ids,rows


def validate_gate(initial,choices,post,summary,*,initial_record,choices_record,post_record):
 """Pure complete-cohort barrier; no object/private arrays read by this gate."""
 ids,rows=validate_prediction_barrier(initial,choices,post,initial_record=initial_record,choices_record=choices_record)
 if summary.get('completed') is not True or summary.get('all_initial_choices_and_postpredictions_sealed_before_private_target_or_label_access') is not True or summary.get('initial_prediction_seal')!=initial_record or summary.get('choices_seal')!=choices_record or summary.get('postprobe_prediction_seal')!=post_record:raise ValueError('Completed root full-evaluation seal required BEFORE selection or targets')
 return ids,rows


def select_records(records):
 """Exactly the four predeclared configurations, repetition0; no error ranking."""
 by={}
 for r in records:
  k=(r['private_configuration_index'],r['repeat_index'])
  if k in by:raise ValueError('Duplicate configuration/repetition')
  by[k]=r
 if set(by)!={(c,r) for c in range(4) for r in range(16)}:raise ValueError('Complete4x16 mapping, no selection/filtering')
 return [by[(s['private_configuration_index'],s['repeat_index'])] for s in SELECTION]


def lattice_edges():
 # POINTS are material probes on a7x3x3 tensor grid, not simulation surface nodes.
 ids=np.arange(63).reshape(7,3,3);edges=[]
 for axis in range(3):
  a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);edges.extend(zip(ids[tuple(a)].ravel(),ids[tuple(b)].ravel()))
 return np.asarray(edges,dtype=np.int64)


def project(points,azimuth_degrees,*,origin_pixels):
 q=np.asarray(points,dtype=float)
 if q.shape!=(63,3) or not np.isfinite(q).all():raise ValueError('Finite exact63 material-point geometry required')
 a=np.deg2rad(azimuth_degrees);e=np.deg2rad(RECIPE['elevation_degrees']);right=np.array([np.cos(a),-np.sin(a),0.]);up=np.array([np.sin(e)*np.sin(a),np.sin(e)*np.cos(a),np.cos(e)])
 d=q-np.array([.12,0.,0.]);return np.asarray(origin_pixels)+RECIPE['world_scale_pixels_per_m']*np.stack([d@right,-d@up],axis=1)


def profile_values(actual,mean,intervals):
 actual=np.asarray(actual,dtype=float);mean=np.asarray(mean,dtype=float);intervals=np.asarray(intervals,dtype=float)
 if actual.shape!=(63,3) or mean.shape!=(63,3) or intervals.shape!=(9,3,3) or not np.isfinite([actual,mean]).all() or not np.isfinite(intervals).all() or np.any(intervals[:,:,0]>intervals[:,:,2]):raise ValueError('Exact finite63mean/9rawquantiles required')
 return dict(actual_z_um=(actual[POINT_IDS,2]-POINTS[POINT_IDS,2])*1e6,mean_z_um=(mean[POINT_IDS,2]-POINTS[POINT_IDS,2])*1e6,lower_z_um=(intervals[:,2,0]-POINTS[POINT_IDS,2])*1e6,upper_z_um=(intervals[:,2,2]-POINTS[POINT_IDS,2])*1e6)


def _font(size):
 from PIL import ImageFont
 p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()


def frame(case,azimuth):
 from PIL import Image,ImageDraw
 im=Image.new('RGB',(RECIPE['width'],RECIPE['height']),'#111820');d=ImageDraw.Draw(im);white='#edf0f2';colors=['#dde5e9','#55b4ef','#f7ac58'];small=_font(16);normal=_font(19);large=_font(25)
 d.text((30,18),f"STATIC F=+1N | unknown E/support | configuration{case['configuration']} / repeat0",font=large,fill=white)
 d.text((30,54),'Camera motion only; true1x geometry. Qualified63-point lattice, not reconstructed surface or temporal forecast.',font=normal,fill='#aabcc9')
 title=['ACTUAL canonical field','PREFIX ONLY prediction',f"AFTER ONE {case['force']:+g}N probe"]
 for col,q in enumerate([case['actual'],case['before'].get('mean'),case['after'].get('mean')]):
  x=col*512;d.text((x+25,110),title[col],font=normal,fill=colors[col]);d.rectangle((x+12,145,x+500,540),outline='#354452')
  if q is None:d.text((x+35,285),'NUMERICAL FAILURE: no admitted mean',font=small,fill='#ff8d8d');continue
  uv=project(q,azimuth,origin_pixels=(x+256,350))
  for a,b in lattice_edges():d.line([tuple(uv[a]),tuple(uv[b])],fill=colors[col],width=1)
  for u,v in uv:d.ellipse((u-2,v-2,u+2,v+2),fill=colors[col])
  if col:
   error=np.sqrt(np.mean((q-case['actual'])**2))*1e6;d.text((x+25,505),f'63-point coordinate RMSE: {error:.3f} um',font=small,fill=white)
 if case['before'].get('mean') is not None and case['after'].get('mean') is not None:
  b=np.mean((case['before']['mean']-case['actual'])**2);a=np.mean((case['after']['mean']-case['actual'])**2)
  d.text((25,557),'AFTER WORSE' if a>b else 'After-probe error <= prefix-only error',font=normal,fill='#ff8d8d' if a>b else '#9cceb8')
 d.text((25,589),'z-displacement profiles [um]; shaded95% CONDITIONAL MODEL bands at9qualifiedpoints only (not calibrated confidence)',font=small,fill=white)
 values=[];profiles={}
 actual=(case['actual'][POINT_IDS,2]-POINTS[POINT_IDS,2])*1e6;values.extend(actual)
 for key in ['before','after']:
  r=case[key]
  if r.get('mean') is not None:
   p=profile_values(case['actual'],r['mean'],r['intervals']);profiles[key]=p
   for k in ['mean_z_um','lower_z_um','upper_z_um']:values.extend(p[k])
 low,high=min(values),max(values);pad=max(1.,.1*(high-low));low-=pad;high+=pad
 for col,y in enumerate([-.04,0.,.04]):
  left=col*512+70;right=col*512+480;top=650;bottom=875
  d.text((col*512+165,624),f'material y={1000*y:+.0f}mm',font=small,fill=white);d.line([(left,top),(left,bottom),(right,bottom)],fill='#bac8d2',width=1)
  for value in np.linspace(low,high,5):
   v=bottom-(value-low)/(high-low)*(bottom-top);d.text((col*512+8,v-8),f'{value:.0f}',font=_font(12),fill='#b3c2cc');d.line([(left,v),(right,v)],fill='#253440')
  ids=np.flatnonzero(np.isclose(POINTS[POINT_IDS,1],y));ids=ids[np.argsort(POINTS[POINT_IDS[ids],0])];xs=left+(POINTS[POINT_IDS[ids],0]-.04)/.16*(right-left)
  yy=lambda v:bottom-(v-low)/(high-low)*(bottom-top)
  d.line([(x,yy(v)) for x,v in zip(xs,actual[ids])],fill=colors[0],width=3)
  for key,color in [('before',colors[1]),('after',colors[2])]:
   if key not in profiles:continue
   p=profiles[key];poly=[(x,yy(v)) for x,v in zip(xs,p['lower_z_um'][ids])]+[(x,yy(v)) for x,v in zip(xs[::-1],p['upper_z_um'][ids][::-1])]
   layer=Image.new('RGBA',im.size,(0,0,0,0));ld=ImageDraw.Draw(layer);ld.polygon(poly,fill=color+'30');im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB');d=ImageDraw.Draw(im);d.line([(x,yy(v)) for x,v in zip(xs,p['mean_z_um'][ids])],fill=color,width=2)
  for x,mm in zip(xs,[40,120,200]):d.text((x-15,bottom+8),str(mm),font=small,fill=white)
  d.text((col*512+150,915),'material x [mm]',font=small,fill=white)
 return im


def _prediction(root,state):
 result=evaluation.restore_prediction(root,state['models']['nonlinear_table']['files'])
 if result.get('completed') is not True or result.get('numerical_controls_passed') is not True:return dict(mean=None,intervals=None,failure=result.get('failures',[]))
 m=result['methods']['adaptive'];q=np.asarray(m['canonical_mean_positions_m']);intervals=np.asarray(m['canonical_position_equal_tail_m'])
 if not m.get('completed') or q.shape!=(63,3) or intervals.shape!=(9,3,3) or not np.isfinite(q).all() or not np.isfinite(intervals).all():raise ValueError('Exact admitted adaptive canonical outputs required')
 return dict(mean=q,intervals=intervals)


def render(recipe_record,initial_record,choices_record,post_record,summary_record,observation_execution_record,*,root=ROOT):
 root=Path(root).resolve();r=io.read_json(root,recipe_record)
 if r.get('recipe')!=RECIPE or r.get('source_sha256')!={n:io.sha(root/n) for n in sources()} or r.get('runtime')!=io.runtime():raise ValueError('Frozen renderer source/recipe/runtime required')
 if set(os.sched_getaffinity(0))!={30,31} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU30/31/noGPU')
 def cap(*args):raise TimeoutError('180s static rendering cap')
 old=signal.signal(signal.SIGALRM,cap);signal.alarm(180)
 try:
  initial=io.read_json(root,initial_record);choices=io.read_json(root,choices_record);post=io.read_json(root,post_record)
  # Before opening the evaluation summary or its private aggregate values,
  # require complete public initial/choice/post metadata first.
  validate_prediction_barrier(initial,choices,post,initial_record=initial_record,choices_record=choices_record)
  summary=io.read_json(root,summary_record);ids,postrows=validate_gate(initial,choices,post,summary,initial_record=initial_record,choices_record=choices_record,post_record=post_record)
  if summary.get('source_sha256')!={n:io.sha(root/n) for n in ['experiments/run_campaign_c38_full_evaluation.py','experiments/run_campaign_c38_prediction_evaluation.py','experiments/campaign_c38_prediction_metrics.py','experiments/campaign_c38_policy_metrics.py']}:raise ValueError('Final evaluation source binding mismatch')
  base={};after={}
  for row,dest,key in [(row,base,row['object_id'])for row in initial['objects']]+[(row,after,(row['object_id'],row['force_Fz_N']))for row in postrows]:
   s=io.read_json(root,row['result'])
   if s.get('completed') is not True or s.get('object_id')!=row['object_id']:raise ValueError('Complete saved object required')
   for model in s['models'].values():
    for f in model['files'].values():io.checked(root,f)
   dest[key]=s
  # Only after ALL prediction files are verified: release selected target members.
  child_summary=io.read_json(root,io.record(root,io.checked(root,summary_record).parent/'initial/summary.json'))
  if child_summary.get('observation_execution')!=observation_execution_record or child_summary.get('initial_prediction_seal')!=initial_record or child_summary.get('choices_seal')!=choices_record:raise ValueError('Evaluation/observation lineage mismatch')
  execution=io.read_json(root,observation_execution_record)
  if execution.get('completed') is not True or {x['object_id'] for x in execution['records']}!=ids:raise ValueError('Same complete64 observation production required')
  selected=select_records(execution['records']);choice={x['object_id']:x['choices']['target_risk'] for x in choices['choices']};out=io.local(root,r['output'])
  if out.exists():raise FileExistsError('Fresh local media only; preserve failures')
  out.mkdir(parents=True);cases=[]
  for row in selected:
   token=row['object_id'];force=float(choice[token])
   with np.load(io.checked(root,row['private_future']),allow_pickle=False) as z:truth=evaluation.extract_truth_members(z)
   if truth['object_id']!=token or truth['private_configuration_index']!=row['private_configuration_index'] or truth['repeat_index']!=0:raise ValueError('Deterministic source-case identity mismatch')
   cases.append(dict(configuration=row['private_configuration_index'],object_id=token,force=force,actual=truth['canonical_target_positions_m'],before=_prediction(root,base[token]),after=_prediction(root,after[(token,force)])))
  import imageio_ffmpeg
  command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f"{RECIPE['width']}x{RECIPE['height']}",'-r',str(RECIPE['fps']),'-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(out/'static-unknown-physics.mp4')]
  process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=(out/'ffmpeg.log').open('wb'),start_new_session=True)
  try:
   for ci,case in enumerate(cases):
    for i,angle in enumerate(np.linspace(*RECIPE['orbit_azimuth_degrees'],RECIPE['frames_per_case'])):
     im=frame(case,float(angle));process.stdin.write(im.tobytes())
     if ci==0 and i==RECIPE['frames_per_case']//2:im.save(out/'preview.png')
   process.stdin.close();code=process.wait(timeout=15)
   if code!=0:raise RuntimeError('Owned ffmpeg failed')
  finally:
   if process.poll() is None:io.cleanup_owned(process,3)
  manifest=dict(completed=True,recipe=recipe_record,initial_prediction_seal=initial_record,choices_seal=choices_record,postprobe_prediction_seal=post_record,completed_full_evaluation=summary_record,selection_rule=SELECTION,selected_objects=[dict(object_id=c['object_id'],configuration=c['configuration'],repeat_index=0,acquired_force_Fz_N=c['force'])for c in cases],fps=30,frame_count=480,geometry_scale=1.,camera_motion_only=True,interval_scope=RECIPE['interval_scope'],continuous_surface_or_RGBD_claim=False,media={n:io.record(root,out/n)for n in ['static-unknown-physics.mp4','preview.png']},raw_media_local=True)
  io.write_new(out/'manifest.json',manifest);return manifest
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','render']);a.add_argument('--recipe',required=True);a.add_argument('--recipe-sha256');a.add_argument('--output')
 for n in ['initial-seal','choices-seal','post-seal','summary','observation-execution']:a.add_argument('--'+n);a.add_argument('--'+n+'-sha256')
 v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.recipe,v.output))
 else:
  rr=lambda name:dict(path=getattr(v,name),sha256=getattr(v,name+'_sha256'));m=render(dict(path=v.recipe,sha256=v.recipe_sha256),rr('initial_seal'),rr('choices_seal'),rr('post_seal'),rr('summary'),rr('observation_execution'));print(json.dumps(dict(completed=m['completed'],frames=m['frame_count'])))
