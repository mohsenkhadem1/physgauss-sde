"""SOURCE ONLY until root review: separate staged surrogate SBC lifecycle.

prepare hashes sources/model inputs and seals seeds BEFORE draws. generate owns
new surrogate labels. infer reads PUBLIC displacement observations only. evaluate
requires every law bundle and all inference PIDs absent before selective truth.
No original C38/C39 labels, measurements, raw trajectories or FEM are used.
"""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback,shutil,resource
def restore_entry_signals():
 """Exec preserves blocked masks; restore owned-controller/worker delivery."""
 signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})


# Restore before NumPy/project imports and any owned work on actual exec entry.
if __name__=='__main__':restore_entry_signals()
import numpy as np
from experiments import campaign_surrogate_sbc as sbc
from experiments.campaign_owned_proc import proc_rss
from experiments import run_campaign_resolved_inference as resolved
io=resolved.io;ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_campaign_surrogate_sbc_v03.py'
DESIGN=dict(schema='MATCHING-NL-SURROGATE-SBC-design-v01',objects=64,phases=list(sbc.PHASES),model='EXACT_STORED_NL_POLYNOMIAL_SURROGATE_NOT_FEM',young_prior='logUniform75000to150000Pa',category_prior=dict(both_ends=.5,separated_bottom_patches=.5),noise_std_m=sbc.NOISE,fixed_action='middle_side_2N',observation_representation='float64 displacement, Horner(c2*x+c1)*x+c0 plus independent normal noise; NO reference positions',target_representation='same Horner displacement, x=75000/E',seeds=dict(latent=2390101,prefix_noise=2390102,SIDE_noise=2390103,randomized_PIT=2390104),RNG='NumPy PCG64 explicit SeedSequence([stream_seed,object_index,phase_index_if_PIT])',coordinate_point_ids=sbc.POINT_IDS.tolist(),coordinate_axes=[0,1,2],PIT_quantities=28,methods=list(sbc.METHODS),gates=sbc.GATES,material_grid_points=17,randomized_equal_tail_probabilities=[.025,.975],coverage_semantics='Randomized equal-tail PIT acceptance; NOT ordinary closed credible-interval coverage at atoms (which may conservatively overcover)',uniformity_tests='NONE; descriptive PIT bins and per-quantity binomial coverage only',histogram_edges=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.],physical_population_calibration=False,active_policy_experiment=False,quantile_roots=False,raw_local=True)
RESOURCES=dict(controller_cores=[10,11],worker_slots=[[i,i+1]for i in range(12,30,2)],threads=2,worker_wall_s=120,global_wall_s=1800,cleanup_s=3,RSS_bytes=2*1024**3,no_GPU=True,reserve_bytes=30*1024**3,projected_write_bytes=1024**3,generator_wall_s=60)
PRIVATE_KEYS={'young_Pa','support_category','target_displacement_m'}
FILES=resolved.FILES


def sources():return sorted(set(resolved.sources())|{SOURCE,'experiments/campaign_owned_proc.py','experiments/run_campaign_surrogate_sbc.py','experiments/campaign_surrogate_sbc.py','tests/test_campaign_surrogate_sbc.py','tests/test_run_campaign_surrogate_sbc_v02.py','experiments/campaign_resolved_quantiles_v04.py','experiments/campaign_cdf_arithmetic.py'})


def prepare(protocol,output,reference_protocol,reference_sha256,*,root=ROOT):
 root=Path(root);dest=io.local(root,protocol)
 if dest.exists()or io.local(root,output).exists():raise FileExistsError('New exclusive SBC namespace only')
 ref=io.read_json(root,dict(path=reference_protocol,sha256=reference_sha256))
 # Hash original PUBLIC model sources without loading model arrays or any truth.
 inputs=ref['inputs']['model_inputs'];model=dict(prefix=inputs['candidate_archive'],SIDE=inputs['side_inputs']['candidate_side_archive'])
 for r in model.values():io.checked(root,r)
 p=dict(schema='SURROGATE-SBC-frozen-v01',status='FROZEN_BEFORE_ANY_GENERATIVE_DRAWS',design=DESIGN,resources=RESOURCES,runtime=io.runtime(),model_inputs=model,source_sha256={n:io.sha(root/n)for n in sources()},output=output,reference_protocol=dict(path=reference_protocol,sha256=reference_sha256))
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='SURROGATE-SBC-frozen-v01'or p.get('status')!='FROZEN_BEFORE_ANY_GENERATIVE_DRAWS'or p.get('design')!=DESIGN or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}:raise ValueError('Exact pre-draw design/source/runtime/resource seal')
 io.checked(root,p['reference_protocol'])
 for r in p['model_inputs'].values():io.checked(root,r)
 io.local(root,p['output']);return p


def rng(stream,index,phase=None):
 words=[DESIGN['seeds'][stream],index]
 if phase is not None:words.append(phase)
 return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def frozen_coefficients(root,p):
 from experiments.campaign_c38_candidate_point_forward import CandidatePointForward,ARCHIVE_KEYS
 from experiments.campaign_c39_side_point_forward import SidePointForward,CompositePointForward,ARCHIVE_KEYS as SIDE_KEYS
 with np.load(io.checked(root,p['model_inputs']['prefix']),allow_pickle=False)as z:
  if set(z.files)!=ARCHIVE_KEYS:raise ValueError('Only frozen public NL candidate nodes')
  a=CandidatePointForward({k:z[k].copy()for k in ARCHIVE_KEYS})
 with np.load(io.checked(root,p['model_inputs']['SIDE']),allow_pickle=False)as z:
  if set(z.files)!=SIDE_KEYS:raise ValueError('Only frozen public SIDE candidate nodes')
  b=SidePointForward({k:z[k].copy()for k in SIDE_KEYS})
 model=CompositePointForward(a,b);coeff={c:dict(prefix=model.observation_coefficients(c,'gravity',sbc.POINT_IDS),SIDE=model.observation_coefficients(c,'middle_side_2N',sbc.POINT_IDS),target=model.observation_coefficients(c,'canonical_up_1N',np.arange(63)))for c in sbc.CATEGORIES};sbc.validate_coefficients(coeff);return coeff


def bundle(root,path,value):
 path.mkdir(parents=True,exist_ok=False);io.save_full_result(path,value);return {n:io.record(root,path/n)for n in FILES}


def read_bundle(root,files):return resolved.load_saved_result(root,files)


def generate(protocol,digest,*,root=ROOT):
 """Future stage, never invoked by source preparation or controls."""
 restore_entry_signals();p=validate(protocol,digest,root=root);out=io.local(root,p['output'])/'generation'
 if out.exists():raise FileExistsError('Never overwrite/retry generation')
 guard(RESOURCES['controller_cores']);out.mkdir(parents=True);coeff=frozen_coefficients(root,p);model=bundle(root,out/'PUBLIC-model',dict(coefficients=coeff));rows=[];start=time.monotonic()
 for i in range(64):
  guard(RESOURCES['controller_cores'])
  if time.monotonic()-start>RESOURCES['generator_wall_s']:raise TimeoutError('Generator60s cap, no retry')
  r=rng('latent',i);E=float(75000.*np.exp(np.log(2.)*r.random()));c=sbc.CATEGORIES[int(r.integers(2))];observed={k:sbc.evaluate_polynomial(coeff[c][k],E)+sbc.NOISE*rng(k+'_noise',i).standard_normal(27)for k in sbc.PHASES}
  public=bundle(root,out/'PUBLIC'/str(i),dict(object_index=i,observed_displacements=observed));private=out/'PRIVATE'/(str(i)+'.npz');private.parent.mkdir(exist_ok=True)
  io.codec.exclusive_npz(private,dict(young_Pa=np.asarray(E,dtype=np.float64),support_category=np.asarray(c),target_displacement_m=sbc.evaluate_polynomial(coeff[c]['target'],E).reshape(63,3)))
  rows.append(dict(object_index=i,public=public,private=io.record(root,private)));io.atomic(out/'generation-journal.json',dict(completed=False,pid=os.getpid(),objects_generated=len(rows),protocol=dict(path=protocol,sha256=digest),raw_local=True))
 seal=dict(schema='SURROGATE-SBC-generation-seal-v01',completed=True,count=64,protocol=dict(path=protocol,sha256=digest),model=model,objects=rows,generator_design_sealed_before_draws=True,raw_local=True)
 io.write_new(out/'seal.json',seal);io.atomic(out/'generation-journal.json',dict(completed=True,pid=os.getpid(),objects_generated=64,protocol=dict(path=protocol,sha256=digest),raw_local=True));return seal


def generation(root,p,protocol_record):
 s=io.read_json(root,io.record(root,io.local(root,p['output'])/'generation/seal.json'))
 if s.get('schema')!='SURROGATE-SBC-generation-seal-v01'or s.get('completed')is not True or s.get('count')!=64 or s.get('protocol')!=protocol_record or s.get('generator_design_sealed_before_draws')is not True or len(s.get('objects',[]))!=64 or {r['object_index']for r in s['objects']}!=set(range(64)):raise ValueError('Complete exact64 pre-designed generation')
 for f in s['model'].values():io.checked(root,f)
 for r in s['objects']:
  for f in r['public'].values():io.checked(root,f)
 return s


def barrier(root,p,protocol_record):
 """No NPZ reads: every PUBLIC law/moment bundle before ANY private member."""
 g=generation(root,p,protocol_record);out=io.local(root,p['output'])/'inference';sr=io.record(root,out/'seal.json');s=io.read_json(root,sr);e=io.read_json(root,s['execution'])
 if s.get('schema')!='SURROGATE-SBC-all-laws-seal-v01'or s.get('completed')is not True or s.get('count')!=64 or s.get('phase_laws')!=128 or s.get('protocol')!=protocol_record or s.get('private_labels_read')is not False or s.get('generation_seal')!=io.record(root,io.local(root,p['output'])/'generation/seal.json'):raise ValueError('Complete128 laws sealed before labels')
 if e.get('completed')is not True or e.get('all_owned_workers_reaped')is not True or e.get('protocol')!=protocol_record or e.get('stage')!='inference'or len(e.get('records',[]))!=64 or {r['object_index']for r in e['records']}!=set(range(64)):raise ValueError('Complete reaped inference execution')
 pids=[e['pid']]+[r['pid']for r in e['records']]
 if len(set(pids))!=65 or any(type(pid)is not int or pid<=1 or Path('/proc',str(pid)).exists()for pid in pids):raise ValueError('All65 inference PIDs must be absent')
 if len(s.get('objects',[]))!=64 or {r['object_index']for r in s['objects']}!=set(range(64)):raise ValueError('Every object law retained')
 by={r['object_index']:r for r in e['records']}
 for r in s['objects']:
  z=by[r['object_index']]
  if z.get('exit_code')!=0 or z.get('reaped')is not True or z.get('files')!=r['files']:raise ValueError('Same complete hash-bound worker law result')
  if set(r['files'])!=FILES:raise ValueError('Exact full saved-law bundle')
  for f in r['files'].values():io.checked(root,f)
  tree=io.read_json(root,r['files']['posterior-structure.json'])
  if tree.get('object_index')!=r['object_index']or tree.get('private_labels_read')is not False or set(tree.get('phases',{}))!=set(sbc.PHASES)or set(tree.get('observations',{}))!=set(sbc.PHASES)or set(tree.get('coefficients',{}))!=set(sbc.CATEGORIES):raise ValueError('Every128 saved phase law and input role must exist before labels')
  for phase in sbc.PHASES:
   q=tree['phases'][phase]
   if q.get('phase')!=phase or type(q.get('completed'))is not bool or type(q.get('numerical_controls_passed'))is not bool or not isinstance(q.get('failures'),list):raise ValueError('Explicit numerical negative or complete phase law')
   if q['completed']and(set(q.get('methods',{}))!=set(sbc.METHODS)or any(m.get('completed')is not True for m in q['methods'].values())):raise ValueError('Complete saved laws require all three integrations')
 # Private file BYTES may be hashed only after all public/law bundles closure.
 for r in g['objects']:io.checked(root,r['private'])
 return g,s,dict(all65_inference_PIDs_absent=True,all128_laws_and_moments_hash_verified_before_private_members=True,prediction_seal=sr)


def extract_truth(z):
 return dict(young_Pa=float(z['young_Pa'].item()),support_category=str(z['support_category'].item()),target_displacement_m=z['target_displacement_m'].copy())


def guard(cores):
 if set(os.sched_getaffinity(0))!=set(cores)or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Declared CPU2/noGPU only')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['RSS_bytes']:raise MemoryError('SBC2GiB RSS cap')
 if shutil.disk_usage(ROOT).free-RESOURCES['projected_write_bytes']<RESOURCES['reserve_bytes']:raise OSError('30GiB reserve after projected output')


def exact_tree_identity(a,b):
 if isinstance(a,np.ndarray)or isinstance(b,np.ndarray):return isinstance(a,np.ndarray)and isinstance(b,np.ndarray)and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
 if isinstance(a,dict)and isinstance(b,dict):return set(a)==set(b)and all(exact_tree_identity(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple))and isinstance(b,(list,tuple)):return len(a)==len(b)and all(exact_tree_identity(x,y)for x,y in zip(a,b))
 if isinstance(a,np.generic):a=a.item()
 if isinstance(b,np.generic):b=b.item()
 if isinstance(a,float)and isinstance(b,float):return np.float64(a).tobytes()==np.float64(b).tobytes()
 return type(a)==type(b)and a==b


def worker(protocol,digest,stage,index,slot,*,root=ROOT):
 restore_entry_signals();p=validate(protocol,digest,root=root);guard(RESOURCES['worker_slots'][slot]);record=dict(path=protocol,sha256=digest)
 if stage=='evaluation':g,sealed,proof=barrier(root,p,record)
 else:g=generation(root,p,record)
 coeff=read_bundle(root,g['model'])['coefficients'];row=next(r for r in g['objects']if r['object_index']==index);public=read_bundle(root,row['public'])
 if public['object_index']!=index or set(public)!= {'object_index','observed_displacements'}:raise ValueError('Public observations only, exact index')
 obs=public['observed_displacements'];out=io.local(root,p['output'])/stage/'objects'/str(index)
 if stage=='inference':
  value=dict(object_index=index,private_labels_read=False,observations=obs,coefficients=coeff,phases={k:sbc.infer(coeff,obs,k)for k in sbc.PHASES})
 else:
  previous=read_bundle(root,next(r for r in sealed['objects']if r['object_index']==index)['files'])
  # Verify exact law inputs from the saved seal before any label member access.
  if not exact_tree_identity(previous['observations'],obs)or not exact_tree_identity(previous['coefficients'],coeff):raise ValueError('Saved law input identity')
  posts={}
  for phase in sbc.PHASES:
   reconstructed,posts[phase]=sbc.compile_inference(coeff,obs,phase)
   if not exact_tree_identity(previous['phases'][phase],reconstructed):raise ValueError('Frozen normalizer/category/material/q63moments reconstruction mismatch BEFORE private labels')
  with np.load(io.checked(root,row['private']),allow_pickle=False)as z:truth=extract_truth(z)
  value=dict(object_index=index,phases={})
  for j,k in enumerate(sbc.PHASES):
   result=sbc.score_laws(coeff,obs,k,truth,rng('randomized_PIT',index,j).random(28),posts=posts[k]);value['phases'][k]=dict(object_index=index,admitted=previous['phases'][k]['numerical_controls_passed']and result['numerical_controls_passed'],evaluation=result)
 return bundle(root,out,value)


def owned_jobs(command_factory,indices,slots,out,*,worker_wall_s,global_wall_s,RSS_bytes,controller_check=lambda:None):
 """Real owned-child supervisor shared by future controller and CPU controls."""
 restore_entry_signals();out=Path(out);queue=list(indices);live={};done=[];start=time.monotonic();failure=None
 state=dict(schema='SURROGATE-SBC-owned-journal-v01',pid=os.getpid(),completed=False,status='STARTING',records=done,pending_indices=queue,live=[])
 def save(status):
  state.update(status=status,wall_s=time.monotonic()-start,live=[dict(object_index=i,pid=proc.pid,slot=slot)for slot,(proc,log,i,beg)in live.items()],pending_indices=list(queue));io.atomic(out/'journal.json',state)
 def interrupted(signum,frame):raise InterruptedError('Owned controller signal '+str(signum))
 old={sig:signal.signal(sig,interrupted)for sig in (signal.SIGTERM,signal.SIGINT)}
 save('STARTING')
 try:
  while queue or live:
   controller_check()
   if time.monotonic()-start>global_wall_s:raise TimeoutError('Global owned SBC stage cap')
   for slot in range(slots):
    if queue and slot not in live:
     i=queue.pop(0);save('LAUNCH_INTENT_'+str(i));log=(out/('worker-'+str(i)+'.log')).open('x')
     # A pending termination cannot land between successful spawn and PID journal.
     mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(old))
     try:
      proc=subprocess.Popen(command_factory(i,slot),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      live[slot]=(proc,log,i,time.monotonic());save('RUNNING')
     except BaseException:log.close();raise
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
   for slot,(proc,log,i,beg)in list(live.items()):
    rss=proc_rss(proc.pid) # Only ENOENT/ESRCH indicate benign process exit.
    if time.monotonic()-beg>worker_wall_s or rss>RSS_bytes:raise RuntimeError('Owned worker wall/RSS budget cap, no retry')
    code=proc.poll()
    if code is not None:
     proc.wait();log.close();done.append(dict(object_index=i,pid=proc.pid,exit_code=code,reaped=True));del live[slot];save('REAPED')
     if code!=0:raise RuntimeError('Worker technical failure, no retry')
   time.sleep(.05)
  state.update(completed=True,all_owned_workers_reaped=True);save('COMPLETE');return done
 except BaseException as error:
  failure=error;state['error']=repr(error);save('CLEANING_FAILURE');raise
 finally:
  # Repeated controller interruption must not cut short owned-group draining.
  for sig in old:signal.signal(sig,signal.SIG_IGN)
  try:
   for slot,(proc,log,i,beg)in list(live.items()):
    code=io.cleanup_owned(proc,3);log.close();done.append(dict(object_index=i,pid=proc.pid,exit_code=code,reaped=True,owned_cleanup=True));del live[slot];save('CLEANING_REAPED')
   state['all_owned_workers_reaped']=not live
   if failure is not None:save('FAILED_ALL_REAPED')
  finally:
   for sig,handler in old.items():signal.signal(sig,handler)


def run_stage(protocol,digest,stage,*,root=ROOT):
 """Bounded owned controller, no retries. Root authorizes later CLI execution."""
 if stage not in ('inference','evaluation'):raise ValueError('Separate staged calls')
 restore_entry_signals();p=validate(protocol,digest,root=root);guard(RESOURCES['controller_cores']);record=dict(path=protocol,sha256=digest)
 if stage=='evaluation':barrier(root,p,record)
 else:generation(root,p,record)
 out=io.local(root,p['output'])/stage;out.mkdir(parents=True,exist_ok=False)
 def command(i,slot):
  env_args=['CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2']
  return ['env',*env_args,'taskset','-c',','.join(map(str,RESOURCES['worker_slots'][slot])),str(ROOT/'.venv/bin/python'),'-m','experiments.run_campaign_surrogate_sbc_v03','worker','--protocol',protocol,'--digest',digest,'--stage',stage,'--index',str(i),'--slot',str(slot)]
 done=owned_jobs(command,range(64),len(RESOURCES['worker_slots']),out,worker_wall_s=RESOURCES['worker_wall_s'],global_wall_s=RESOURCES['global_wall_s'],RSS_bytes=RESOURCES['RSS_bytes'],controller_check=lambda:guard(RESOURCES['controller_cores']))
 for r in done:r['files']={n:io.record(root,out/'objects'/str(r['object_index'])/n)for n in FILES}
 journal=io.record(root,out/'journal.json');execution=dict(completed=True,pid=os.getpid(),protocol=record,stage=stage,all_owned_workers_reaped=True,records=done,journal=journal);io.write_new(out/'execution.json',execution)
 if stage=='inference':
  seal=dict(schema='SURROGATE-SBC-all-laws-seal-v01',completed=True,count=64,phase_laws=128,protocol=record,generation_seal=io.record(root,io.local(root,p['output'])/'generation/seal.json'),execution=io.record(root,out/'execution.json'),objects=[dict(object_index=r['object_index'],files=r['files'])for r in done],private_labels_read=False)
  io.write_new(out/'seal.json',seal)
 else:
  values=[read_bundle(root,r['files'])for r in done];summary=dict(completed=True,protocol=record,phases={k:sbc.aggregate([v['phases'][k]for v in values])for k in sbc.PHASES},physical_population_calibration=False,independent_objects=64,phase_laws=128,coordinates_independent=False,randomized_equal_tail_coverage=True,raw_local=True);io.write_new(out/'summary.json',io.jsonable(summary))
 return execution


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','generate','inference','evaluation','worker'));a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');a.add_argument('--reference-protocol');a.add_argument('--reference-sha256');a.add_argument('--stage',choices=('inference','evaluation'));a.add_argument('--index',type=int);a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,v.reference_protocol,v.reference_sha256))
 elif v.mode=='generate':generate(v.protocol,v.digest)
 elif v.mode=='worker':worker(v.protocol,v.digest,v.stage,v.index,v.slot)
 else:run_stage(v.protocol,v.digest,v.mode)
