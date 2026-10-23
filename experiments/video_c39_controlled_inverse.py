"""C39 static actual/before/one-selected-probe visualization, not forecasting.

Deterministic four configurations/repetition0; no error-based selection. Freeze
source/recipe and all provenance before selected values. No acquired DOWN/SIDE
observation arrays are read. Raw media local; ideal63-point lattice only.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,os,resource,signal,subprocess
import numpy as np
from experiments import video_c38_controlled_inverse as old
from experiments import run_campaign_c39_full_evaluation as full
io=full.io;evaluation=full.evaluation;ROOT=Path(__file__).resolve().parents[1]
POINTS=old.POINTS;POINT_IDS=old.POINT_IDS
project=old.project;lattice_edges=old.lattice_edges;profile_values=old.profile_values;_font=old._font
SOURCE='experiments/video_c39_controlled_inverse.py'
SELECTION=old.SELECTION
RECIPE={**old.RECIPE,'schema':'C39-static-typed-unknown-physics-point-video-v01','after_policy':'target_risk','interval_scope':'Raw z-only 95% model-conditional equal-tail intervals at9qualifiedtop points. Original10nm absolutequantile stopping retained; no nominalCDFaccuracy/calibration claim for tinyspread coordinates. Sensornoise NOT added to latentfield.', 'purpose':'Static canonical+1 load prediction before/after ONE typedprobe; cameraorbit only, NOT a temporal forecast or learned continuoussurface.'}
INPUTS=dict(initial=dict(path='runs/gaussian-campaign-v01/c38-initial-public-inference-v03/prediction-seal.json',sha256='4b8c8dbaed0a566d4cfd72a51a60921f2384be8b78c29e2b20d416b32fdc91e7'),choices=dict(path='runs/gaussian-campaign-v01/c39-typed-static-policy-v01/choices-seal.json',sha256='537d4bd6c7bfe550e53152f9505b8443e9dd74f5d155d3fb29caf4fe63ff5894'),post=dict(path='runs/gaussian-campaign-v01/c39-SIDE-postprobe-inference-v01/prediction-seal.json',sha256='12538f5b2d2aa5b1af390c25b3830978a97eea5b49a0071ba9ca16e3315a131d'),summary=dict(path='runs/gaussian-campaign-v01/c39-full-evaluation-v01/summary.json',sha256='a6aa1a8838d27cd39c7c87984ad92d9b821c0315311caae02a12b6bcdb3c447c'),reference=dict(path='runs/operator-dev/c39-full-evaluation-source-v01/reference.json',sha256='d22f471d2797a8b1b7775021af1963b72a4a75c75393a67970e7f58a7eb2b33c'),observation_execution=dict(path='runs/gaussian-campaign-v01/c38-controlled-observations-v01/PRIVATE/execution.json',sha256='94853b220696bcbced7ae48c9d42cf6a37c87d4d28de510f3030093a53888cbe'))
PROTOCOLS=dict(initial=dict(path='runs/operator-dev/c38-initial-public-inference-v03/protocol-v03.json',sha256='cd2238db8cd196f289d5a85f976ad7011f02a01e665a97872a0bcf1a7fd893e0'),choices=dict(path='runs/operator-dev/c39-typed-static-policy-execution-v01/protocol.json',sha256='821e3d062db0f9a831e28825a7e927d0c2b3f7cd91bd71e1fdca68edbd1fd566'),post=dict(path='runs/operator-dev/c39-SIDE-postprobe-inference-v01/protocol.json',sha256='c94ed9e8fe09bbf50cd13d6075b41b8fbff2d04f9a81577460b0c779561e56b3'))


def sources():return sorted(set(old.sources())|set(full.source_records(ROOT))|{SOURCE,'tests/test_video_c39_controlled_inverse.py'})


def verify_protocol_sources(root):
 for record in PROTOCOLS.values():
  p=io.read_json(root,record)
  for name,digest in p['source_sha256'].items():io.checked(root,dict(path=name,sha256=digest))


def prepare(recipe_path,output,*,root=ROOT):
 root=Path(root).resolve();p=io.local(root,recipe_path)
 if p.exists()or io.local(root,output).exists():raise FileExistsError('Freshrecipe/output; retainfailures')
 # Hash records only, never extract trajectory, prediction or target members.
 for record in INPUTS.values():io.checked(root,record)
 verify_protocol_sources(root)
 p.parent.mkdir(parents=True,exist_ok=True);io.write_new(p,dict(recipe=RECIPE,inputs=INPUTS,protocols=PROTOCOLS,output=output,source_sha256={n:io.sha(root/n)for n in sources()},runtime=io.runtime(),status='FROZEN_SELECTION_BEFORE_SELECTED_VALUE_READS',actual_selected_values_read=False));return io.sha(p)


def validate_prediction_barrier(initial,choices,post,*,initial_record,choices_record):
 if initial.get('completed')is not True or initial.get('count')!=64 or initial.get('model_executions')!=128 or initial.get('private_truth_access')is not False:raise ValueError('Completeinitial64x2')
 ids={r['object_id']for r in initial.get('objects',[])}
 if len(ids)!=64 or len(initial.get('objects',[]))!=64:raise ValueError('All64exactidentities')
 if choices.get('schema')!='C39-typed-static-policy-choices-seal-v01'or choices.get('completed')is not True or choices.get('count')!=64 or choices.get('model_executions')!=128 or choices.get('initial_prediction_seal')!=initial_record or choices.get('extension_design')!=full.DESIGN or choices.get('all_choices_sealed_before_SIDE_noise')is not True:raise ValueError('Alltyped64x2choices beforeSIDE')
 if any(choices.get(k)is not False for k in ('private_SIDE_outcomes_access','acquired_DOWN_observations_or_posteriors_access','private_target_or_latent_access')):raise ValueError('Choice informationbarrier')
 if len(choices.get('choices',[]))!=64 or {r['object_id']for r in choices['choices']}!=ids or set(choices.get('model_choices',{}))!=set(io.MODELS):raise ValueError('Exact64two-model choicejoin')
 for rows in choices['model_choices'].values():
  if len(rows)!=64 or {r['object_id']for r in rows}!=ids:raise ValueError('All64model choices')
  for row in rows:
   c=row['choices']
   if set(c)!=set(full.metrics.POLICIES)or c['fixed_down']!=full.metrics.DOWN or c['fixed_side']!=full.metrics.SIDE or any(a not in (full.metrics.DOWN,full.metrics.SIDE)for a in c.values()):raise ValueError('Exacttypedactions, noFzalias')
 if post.get('schema')!='C39-public-SIDE-postprobe-prediction-seal-v01'or post.get('completed')is not True or post.get('count')!=64 or post.get('model_executions')!=128 or post.get('initial_prediction_seal')!=initial_record or post.get('choices_seal')!=choices_record or post.get('private_target_or_latent_access')is not False or post.get('all_postpredictions_sealed_before_private_evaluation')is not True:raise ValueError('CompleteSIDE64x2beforevalues')
 rows=post.get('objects',[])
 if len(rows)!=64 or {r['object_id']for r in rows}!=ids or any(r.get('action_id')!=full.metrics.SIDE for r in rows):raise ValueError('EverytypedSIDEposterior required')
 return ids


def validate_gate(initial,choices,post,summary,*,initial_record,choices_record,post_record,reference_record):
 ids=validate_prediction_barrier(initial,choices,post,initial_record=initial_record,choices_record=choices_record)
 if summary.get('completed')is not True or summary.get('choices_seal')!=choices_record or summary.get('SIDE_prediction_seal')!=post_record or summary.get('frozen_reference')!=reference_record or summary.get('design')!=full.DESIGN or summary.get('all_SIDE_predictions_sealed_before_SIDE_dependent_scoring')is not True:raise ValueError('CompletedboundC39evaluation beforeselection/targets')
 return ids


def select_records(records):return old.select_records(records)


def action_label(action_id):
 labels={full.metrics.DOWN:'DOWN -2 N along z',full.metrics.SIDE:'SIDE +2 N along y'}
 if action_id not in labels:raise ValueError('TypedDOWN/SIDE required; no scalarFz alias')
 return labels[action_id]


def frame(case,azimuth):
 label=action_label(case['action_id'])
 from PIL import Image,ImageDraw
 im=Image.new('RGB',(RECIPE['width'],RECIPE['height']),'#111820');d=ImageDraw.Draw(im);white='#edf0f2';colors=['#dde5e9','#55b4ef','#f7ac58'];small=_font(16);normal=_font(19);large=_font(25)
 d.text((30,18),f"STATIC canonical +1 N | unknown stiffness/supports | configuration {case['configuration']} / repeat 0",font=large,fill=white)
 d.text((30,54),'Static fields; camera orbit only; true 1x geometry. 63-point lattice, not a reconstructed surface or temporal forecast.',font=normal,fill='#aabcc9')
 d.text((30,82),'Raw z intervals: conditional model only; original 10 nm quantile resolution, not calibrated confidence.',font=small,fill='#aabcc9')
 title=['ACTUAL canonical field','BEFORE POKE / prefix only',f'AFTER ONE {label}']
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
 d.text((25,589),'z-displacement profiles [um]; shaded raw 95% CONDITIONAL MODEL bands at 9 qualified points only',font=small,fill=white)
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



def guard(root):
 if set(os.sched_getaffinity(0))!={30,31}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU30/31, thread2, noGPU')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*1024**3:raise MemoryError('Renderer2GiB cap')


def render(recipe_record,*,root=ROOT):
 root=Path(root).resolve();r=io.read_json(root,recipe_record)
 if r.get('recipe')!=RECIPE or r.get('inputs')!=INPUTS or r.get('protocols')!=PROTOCOLS or r.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or r.get('runtime')!=io.runtime():raise ValueError('Frozenrecipe/source/runtime/lineage')
 guard(root);verify_protocol_sources(root)
 def cap(*args):raise TimeoutError('C39staticrender180scap')
 previous=signal.signal(signal.SIGALRM,cap);signal.alarm(180)
 try:
  initial=io.read_json(root,INPUTS['initial']);choices=io.read_json(root,INPUTS['choices']);post=io.read_json(root,INPUTS['post'])
  validate_prediction_barrier(initial,choices,post,initial_record=INPUTS['initial'],choices_record=INPUTS['choices'])
  if initial.get('protocol_sha256')!=PROTOCOLS['initial']['sha256']or choices.get('protocol')!=PROTOCOLS['choices']or post.get('protocol')!=PROTOCOLS['post']:raise ValueError('Exactsealed runtimeprotocols')
  reference=io.read_json(root,INPUTS['reference'])
  if reference.get('schema')!='C39-fixed-metric-reference-v01'or reference.get('design')!=full.DESIGN or reference.get('source_sha256')!=full.source_records(root)or reference.get('initial_prediction_seal')!=INPUTS['initial']:raise ValueError('Exactreference/source binding')
  # Hash all initial/DOWN/SIDE bundles before opening the evaluation summary,
  # choosing cases or extracting ANY private target/prediction array members.
  ids,verified_choices,side=full.barrier(root,reference,INPUTS['choices'],INPUTS['post'])
  summary=io.read_json(root,INPUTS['summary']);validate_gate(initial,choices,post,summary,initial_record=INPUTS['initial'],choices_record=INPUTS['choices'],post_record=INPUTS['post'],reference_record=INPUTS['reference'])
  if summary.get('source_sha256')!=full.source_records(root):raise ValueError('Completedmetric source mismatch')
  execution=io.read_json(root,INPUTS['observation_execution'])
  if execution.get('completed')is not True or len(execution.get('records',[]))!=64 or {x['object_id']for x in execution['records']}!=ids:raise ValueError('Samecomplete64identity/targetsource')
  selected=select_records(execution['records']);initialstates={x['object_id']:io.read_json(root,x['result'])for x in initial['objects']};oldpost=io.read_json(root,reference['C38_postseal']);down={x['object_id']:io.read_json(root,x['result'])for x in oldpost['objects']if x['force_Fz_N']==-2.};choice={x['object_id']:x['choices']['target_risk']for x in choices['model_choices']['nonlinear_table']};primary={x['object_id']:x['choices']['target_risk']for x in choices['choices']}
  if choice!=primary:raise ValueError('Primary/nonlinearmodel target-risk choices differ')
  out=io.local(root,r['output'])
  if out.exists():raise FileExistsError('Freshlocalmedia; failurespreserved')
  out.mkdir(parents=True);cases=[]
  for row in selected:
   guard(root);token=row['object_id'];action=choice[token];action_label(action)
   with np.load(io.checked(root,row['private_future']),allow_pickle=False)as z:truth=evaluation.extract_truth_members(z)
   if truth['object_id']!=token or truth['private_configuration_index']!=row['private_configuration_index']or truth['repeat_index']!=0:raise ValueError('Exactfixedcase/repeat0source')
   after=down[token]if action==full.metrics.DOWN else side[token]
   if after.get('object_id')!=token:raise ValueError('Selectedprediction identity')
   cases.append(dict(configuration=row['private_configuration_index'],object_id=token,action_id=action,actual=truth['canonical_target_positions_m'],before=old._prediction(root,initialstates[token]),after=old._prediction(root,after)))
  import imageio_ffmpeg
  cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f"{RECIPE['width']}x{RECIPE['height']}",'-r',str(RECIPE['fps']),'-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(out/'static-unknown-physics.mp4')]
  with(out/'ffmpeg.log').open('wb')as log:
   process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,start_new_session=True)
   try:
    for ci,case in enumerate(cases):
     for i,angle in enumerate(np.linspace(*RECIPE['orbit_azimuth_degrees'],RECIPE['frames_per_case'])):
      guard(root);image=frame(case,float(angle));process.stdin.write(image.tobytes())
      if ci==0 and i==RECIPE['frames_per_case']//2:image.save(out/'preview.png')
    process.stdin.close();code=process.wait(timeout=15)
    if code!=0:raise RuntimeError('Ownedffmpegfailed')
   finally:
    if process.poll()is None:io.cleanup_owned(process,3)
  manifest=dict(completed=True,recipe=recipe_record,inputs=INPUTS,protocols=PROTOCOLS,selection_rule=SELECTION,selected_objects=[dict(object_id=c['object_id'],configuration=c['configuration'],repeat_index=0,selected_action_id=c['action_id'],force_world_N=([0.,0.,-2.]if c['action_id']==full.metrics.DOWN else[0.,2.,0.]))for c in cases],fps=30,frame_count=480,geometry_scale=1.,camera_motion_only=True,temporal_forecast=False,interval_scope=RECIPE['interval_scope'],continuous_surface_or_RGBD_claim=False,raw_media_local=True,media={n:io.record(root,out/n)for n in ('static-unknown-physics.mp4','preview.png')})
  io.write_new(out/'manifest.json',manifest);return manifest
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','render'));p.add_argument('--recipe',required=True);p.add_argument('--recipe-sha256');p.add_argument('--output');v=p.parse_args()
 if v.mode=='prepare':print(prepare(v.recipe,v.output))
 else:r=render(dict(path=v.recipe,sha256=v.recipe_sha256));print(json.dumps(dict(completed=r['completed'],frames=r['frame_count'])))
