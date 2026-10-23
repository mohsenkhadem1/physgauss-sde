"""Evaluation-only continuation on immutable v02 cohort/laws; no new draws.

Preserves the technical stop, interrupted/unlaunched records and complete old
scores. Reuses complete bundles by hash, computes only remaining same-cohort
scores with the unchanged pre-label reconstruction and frozen PIT streams.
"""
from pathlib import Path
import argparse,json,os,signal,time
signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
import numpy as np
from experiments import run_campaign_surrogate_sbc_v02 as old
from experiments import run_campaign_surrogate_sbc_v03 as fixed
io=old.io;sbc=old.sbc;ROOT=old.ROOT;FILES=old.FILES;RESOURCES=old.RESOURCES
SOURCE='experiments/run_campaign_surrogate_sbc_evaluation_continuation.py'
ORIGINAL=dict(path='runs/operator-dev/matching-prior-surrogate-sbc-v02/protocol.json',sha256='8b641efeda802561471f55f0a4311036a5290ed37030ef610c2c2dd18814aee6')


def sources():return sorted(set(old.sources())|{SOURCE,'experiments/run_campaign_surrogate_sbc_v03.py','experiments/campaign_owned_proc.py','experiments/supervise_surrogate_sbc_continuation.py','tests/test_surrogate_sbc_evaluation_continuation.py'})


def completed_score(value,index):
 if value.get('object_index')!=index or set(value.get('phases',{}))!=set(sbc.PHASES):raise ValueError('Complete same-object two-phase score bundle')
 for phase,row in value['phases'].items():
  if row.get('object_index')!=index or type(row.get('admitted'))is not bool:raise ValueError('Explicit negative/positive score admission')
  e=row.get('evaluation',{})
  if set(e.get('methods',{}))!=set(sbc.METHODS)or type(e.get('numerical_controls_passed'))is not bool:raise ValueError('All methods including numerical failures retained')
  for method,r in e['methods'].items():
   if type(r.get('completed'))is not bool or(r['completed']and len(r.get('scores',[]))!=28):raise ValueError('Every method complete or explicit numerical negative')
 return value


def plan(root):
 p=old.validate(ORIGINAL['path'],ORIGINAL['sha256'],root=root);g,sealed,proof=old.barrier(root,p,ORIGINAL);out=io.local(root,p['output'])/'evaluation';jr=io.record(root,out/'journal.json');j=io.read_json(root,jr)
 if j.get('completed')is not False or j.get('status')!='FAILED_ALL_REAPED'or j.get('all_owned_workers_reaped')is not True or j.get('live')!=[]:raise ValueError('Preserve exact stopped/reaped evaluation history')
 rows=j.get('records',[]);indices=[r['object_index']for r in rows];pending=j.get('pending_indices',[])
 if len(indices)!=len(set(indices))or len(pending)!=len(set(pending))or set(indices)&set(pending)or set(indices)|set(pending)!=set(range(64)):raise ValueError('All64 original partial/cancelled/unlaunched identities retained')
 pids=[j['pid']]+[r['pid']for r in rows]
 if any(type(pid)is not int or Path('/proc',str(pid)).exists()for pid in pids)or any(r.get('reaped')is not True for r in rows):raise ValueError('All old evaluation processes absent/reaped')
 reused=[]
 for i in range(64):
  directory=out/'objects'/str(i);existing=[n for n in FILES if(directory/n).exists()]
  if len(existing)==len(FILES):
   files={n:io.record(root,directory/n)for n in FILES};completed_score(old.read_bundle(root,files),i);reused.append(dict(object_index=i,origin='REUSED_COMPLETE_V02',files=files))
  elif existing:raise ValueError('Partially written old bundle requires separate review, never overwrite')
 remaining=sorted(set(range(64))-{r['object_index']for r in reused})
 return dict(original_protocol=ORIGINAL,generation_seal=io.record(root,io.local(root,p['output'])/'generation/seal.json'),inference_seal=proof['prediction_seal'],original_partial_journal=jr,original_technical_stop=io.record(root,root/'runs/operator-dev/matching-prior-surrogate-sbc-v02/technical-stop-receipt.json'),original_supervisor_failure=io.record(root,root/'runs/operator-dev/matching-prior-surrogate-sbc-v02/actual-evaluation-receipt.json'),original_partial_records=rows,original_unlaunched_indices=pending,reused=reused,remaining_indices=remaining,denominators=dict(objects=64,reused_complete_objects=len(reused),remaining_objects=len(remaining),old_interrupted_records=sum(bool(r.get('owned_cleanup'))for r in rows),old_unlaunched_objects=len(pending)))


def prepare(protocol,output,*,root=ROOT):
 dest=io.local(root,protocol)
 if dest.exists()or io.local(root,output).exists():raise FileExistsError('New evaluation-only namespace')
 p=dict(schema='SURROGATE-SBC-evaluation-only-continuation-v01',status='FROZEN_BEFORE_CONTINUATION',inputs=plan(root),design=old.DESIGN,resources=RESOURCES,runtime=io.runtime(),source_sha256={n:io.sha(root/n)for n in sources()},output=output,new_generation_or_inference=False,raw_local=True)
 dest.parent.mkdir(parents=True,exist_ok=True);io.write_new(dest,p);return io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 p=io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='SURROGATE-SBC-evaluation-only-continuation-v01'or p.get('status')!='FROZEN_BEFORE_CONTINUATION'or p.get('design')!=old.DESIGN or p.get('resources')!=RESOURCES or p.get('runtime')!=io.runtime()or p.get('source_sha256')!={n:io.sha(root/n)for n in sources()}or p.get('new_generation_or_inference')is not False or p.get('inputs')!=plan(root):raise ValueError('Exact immutable same-cohort evaluation-only continuation')
 io.local(root,p['output']);return p


def worker(protocol,digest,index,slot,*,root=ROOT):
 old.restore_entry_signals();p=validate(protocol,digest,root=root);old.guard(RESOURCES['worker_slots'][slot])
 if index not in p['inputs']['remaining_indices']:raise ValueError('Only frozen remaining indices can be evaluated')
 original=old.validate(ORIGINAL['path'],ORIGINAL['sha256'],root=root);g,sealed,proof=old.barrier(root,original,ORIGINAL)
 coeff=old.read_bundle(root,g['model'])['coefficients'];row=next(r for r in g['objects']if r['object_index']==index);public=old.read_bundle(root,row['public'])
 if public['object_index']!=index or set(public)!= {'object_index','observed_displacements'}:raise ValueError('Public observations only, exact index')
 obs=public['observed_displacements'];out=io.local(root,p['output'])/'objects'/str(index)
 previous=old.read_bundle(root,next(r for r in sealed['objects']if r['object_index']==index)['files'])
 if not old.exact_tree_identity(previous['observations'],obs)or not old.exact_tree_identity(previous['coefficients'],coeff):raise ValueError('Saved law input identity')
 posts={}
 for phase in sbc.PHASES:
  reconstructed,posts[phase]=sbc.compile_inference(coeff,obs,phase)
  if not old.exact_tree_identity(previous['phases'][phase],reconstructed):raise ValueError('Frozen normalizer/category/material/q63moments reconstruction mismatch BEFORE private labels')
 with np.load(io.checked(root,row['private']),allow_pickle=False)as z:truth=old.extract_truth(z)
 value=dict(object_index=index,phases={})
 for j,k in enumerate(sbc.PHASES):
  result=sbc.score_laws(coeff,obs,k,truth,old.rng('randomized_PIT',index,j).random(28),posts=posts[k]);value['phases'][k]=dict(object_index=index,admitted=previous['phases'][k]['numerical_controls_passed']and result['numerical_controls_passed'],evaluation=result)
 return old.bundle(root,out,value)


def merged_rows(root,p,new_rows):
 expected=p['inputs']['remaining_indices']
 if len(new_rows)!=len(expected)or {r['object_index']for r in new_rows}!=set(expected)or any(r.get('exit_code')!=0 or r.get('reaped')is not True for r in new_rows):raise ValueError('Every remaining worker complete/reaped, no subset')
 rows=p['inputs']['reused']+[dict(object_index=r['object_index'],origin='CONTINUED_SAME_COHORT',files=r['files'])for r in new_rows]
 if len(rows)!=64 or {r['object_index']for r in rows}!=set(range(64)):raise ValueError('Exactly64 original objects, failures retained')
 values=[]
 for row in sorted(rows,key=lambda r:r['object_index']):
  value=completed_score(old.read_bundle(root,row['files']),row['object_index']);values.append(value)
 return rows,values


def run(protocol,digest,*,root=ROOT):
 p=validate(protocol,digest,root=root);old.guard(RESOURCES['controller_cores']);out=io.local(root,p['output']);out.mkdir(parents=True,exist_ok=False)
 def command(i,slot):
  return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=2','taskset','-c',','.join(map(str,RESOURCES['worker_slots'][slot])),str(ROOT/'.venv/bin/python'),'-m','experiments.run_campaign_surrogate_sbc_evaluation_continuation','worker','--protocol',protocol,'--digest',digest,'--index',str(i),'--slot',str(slot)]
 done=fixed.owned_jobs(command,p['inputs']['remaining_indices'],len(RESOURCES['worker_slots']),out,worker_wall_s=RESOURCES['worker_wall_s'],global_wall_s=RESOURCES['global_wall_s'],RSS_bytes=RESOURCES['RSS_bytes'],controller_check=lambda:old.guard(RESOURCES['controller_cores']))
 for r in done:r['files']={n:io.record(root,out/'objects'/str(r['object_index'])/n)for n in FILES}
 rows,values=merged_rows(root,p,done);execution=dict(completed=True,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),all_owned_workers_reaped=True,records=done,journal=io.record(root,out/'journal.json'),original_partial_journal=p['inputs']['original_partial_journal']);io.write_new(out/'execution.json',execution)
 summary=dict(completed=True,protocol=dict(path=protocol,sha256=digest),original_generation_seal=p['inputs']['generation_seal'],original_inference_seal=p['inputs']['inference_seal'],objects=64,reused_objects=len(p['inputs']['reused']),continued_objects=len(done),phases={k:sbc.aggregate([v['phases'][k]for v in values])for k in sbc.PHASES},physical_population_calibration=False,coordinates_independent=False,randomized_equal_tail_coverage=True,new_generation_or_inference=False,raw_local=True)
 io.write_new(out/'merged-score-seal.json',dict(completed=True,count=64,protocol=dict(path=protocol,sha256=digest),objects=rows,original_partial_journal=p['inputs']['original_partial_journal'],all_original_partial_records_preserved=True));io.write_new(out/'summary.json',io.jsonable(summary));return summary


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run','worker'));a.add_argument('--protocol',required=True);a.add_argument('--digest');a.add_argument('--output');a.add_argument('--index',type=int);a.add_argument('--slot',type=int);v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 elif v.mode=='worker':worker(v.protocol,v.digest,v.index,v.slot)
 else:run(v.protocol,v.digest)
