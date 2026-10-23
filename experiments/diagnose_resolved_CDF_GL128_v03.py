"""Opt-in public-only trace of ONE failed linear PREFIX adaptive CDF.

No production module mutation; exact v02 guard and GL128 only. Raw central estimates and
bounds are returned unchanged to the immutable v02 root routine. No truth,
noise generation, policy decisions or private observation inputs are permitted.
"""
from pathlib import Path
import argparse,json,math,os,signal,time
import numpy as np
from experiments import run_campaign_resolved_inference as run
from experiments import campaign_resolved_prefix_inference_v02 as prefix
from experiments import campaign_resolved_quantiles_v02 as quantile
from experiments.campaign_c38_postprobe_memo import _global_target_bounds,coordinate_cdf_bounds
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/diagnose_resolved_CDF_GL128_v03.py'
RESOURCES=dict(CPU_cores=[30,31],CPU_threads=2,RSS_bytes=2*1024**3,wall_s=120,no_CUDA=True)


def sources():return sorted(set(run.sources())|{SOURCE,'tests/test_diagnose_resolved_CDF_GL128_v03.py'})


def CDF_conditions(row):
 vals={k:row.get(k)for k in ('cdf','lower','upper')};finite=all(isinstance(x,(int,float,np.number))and math.isfinite(x)for x in vals.values())
 return dict(finite=finite,bounds_in_unit_interval=bool(finite and 0<=vals['lower']<=vals['upper']<=1),central_above_lower=bool(finite and vals['lower']<=vals['cdf']),central_below_upper=bool(finite and vals['cdf']<=vals['upper']))


def trace_query(cdf,trace,*,left_limit=False):
 def query(x):
  row=cdf(x);conditions=CDF_conditions(row);entry=dict(value_m=float(x),left_limit=left_limit,raw={k:(float(row[k])if math.isfinite(row[k])else repr(float(row[k])))for k in ('cdf','lower','upper')},conditions=conditions)
  trace['last_queries']=(trace['last_queries']+[entry])[-16:];trace['query_count']+=1
  if not all(conditions.values()):trace['offending_queries'].append(entry)
  return row
 return query


def quantile_support_compare(value,bound):
 from experiments.campaign_c38_polynomial_likelihood_v02 import rational
 return int(bool(rational(value)>bound))-int(bool(rational(value)<bound))


def trace_coordinate(post,targets,probability,trace):
 lo,hi=_global_target_bounds(targets,post)
 if lo==hi:return dict(kind='common_constant_atom')
 def raw_cdf(x,*,left_limit=False):
  row=coordinate_cdf_bounds(post,targets,x,left_limit=left_limit)
  if not all(CDF_conditions(row).values()):
   raw=post.coordinate_cdf(targets,float(x),maximum_root_mass=1e-8)
   trace['underlying_CDF']={k:v for k,v in raw.items()if k!='root_boxes'}
   trace['support_relation']=dict(below_exact_support=bool(quantile_support_compare(x,lo)<0),above_exact_support=bool(quantile_support_compare(x,hi)>0),Z=float(post.Z),Z_error_estimate=float(post.Z_error_estimate),categories=len(post.categories))
  return row
 cdf=trace_query(raw_cdf,trace)
 left=trace_query(lambda x:raw_cdf(x,left_limit=True),trace,left_limit=True)
 for value in sorted({float(c[0])for c in targets.values()if c[1]==0 and c[2]==0}):
  L,H=left(value),cdf(value)
  if L['upper']<probability<=H['lower']:return dict(kind='category_mixture_atom')
 # Identical unchanged v02 routine/default tolerances: logging only.
 return quantile.bounded_continuous_quantile(cdf,np.nextafter(float(lo),-np.inf),np.nextafter(float(hi),np.inf),probability)


def prepare(protocol,output,*,resolved_protocol,resolved_protocol_sha256,failed_job_id,root=ROOT):
 root=Path(root).resolve();dest=run.io.local(root,protocol)
 if dest.exists()or run.io.local(root,output).exists():raise FileExistsError('Fresh exclusive trace protocol/output')
 rp=dict(path=resolved_protocol,sha256=resolved_protocol_sha256);p=run.validate(resolved_protocol,resolved_protocol_sha256,root=root);job=next((j for j in p['inputs']['jobs']if j['id']==failed_job_id),None)
 if job is None or job['kind']!='prefix':raise ValueError('Exactly one declared failed PREFIX')
 execution=run.io.read_json(root,run.io.record(root,run.io.local(root,p['output']+'/execution.json')));rows=[r for r in execution['records']if r['job_id']==failed_job_id]
 if len(rows)!=1 or rows[0].get('completed')is not True or rows[0].get('reaped')is not True:raise ValueError('Failed numerical arm must be complete/reaped, never inspect live state')
 result=run.output_result(root,rows[0],resolved_protocol_sha256);model=result['models']['linear_rest_tangent'];numeric=run.io.read_json(root,model['files']['numerics.json'])
 if model.get('numerical_controls_passed')is not False or not any(e.get('method')=='GL128'and e.get('stage')=='CANONICAL_QUANTILES'and 'Finite ordered numerical CDF enclosure required'in e.get('error','')for e in numeric.get('failures',[])):raise ValueError('Exact observed adaptive linear-prefix CDF guard failure required')
 recipe=dict(schema='RESOLVED-CDF-public-failed-prefix-GL128-trace-v03',status='FROZEN_BEFORE_SINGLE_PUBLIC_TRACE',resolved_protocol=rp,failed_job_id=failed_job_id,failed_result=rows[0]['result'],failed_model_files=model['files'],output=output,source_sha256={n:run.io.sha(root/n)for n in sources()},runtime=run.io.runtime(),resources=RESOURCES,only_model='linear_rest_tangent',only_method='GL128',only_v02_guard_calls=True,prior_likelihood_observations_and_old_outputs_unchanged=True,private_truth_access=False,retries=0)
 dest.parent.mkdir(parents=True,exist_ok=True);run.io.write_new(dest,recipe);return run.io.sha(dest)


def validate(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p=run.io.read_json(root,dict(path=protocol,sha256=digest))
 if p.get('schema')!='RESOLVED-CDF-public-failed-prefix-GL128-trace-v03'or p.get('status')!='FROZEN_BEFORE_SINGLE_PUBLIC_TRACE'or p.get('resources')!=RESOURCES or p.get('runtime')!=run.io.runtime()or p.get('source_sha256')!={n:run.io.sha(root/n)for n in sources()}or p.get('private_truth_access')is not False or p.get('only_v02_guard_calls')is not True or p.get('only_model')!='linear_rest_tangent'or p.get('only_method')!='GL128'or p.get('retries')!=0:raise ValueError('Exact public one-case trace contract')
 rp=p['resolved_protocol'];old=run.validate(rp['path'],rp['sha256'],root=root);job=next(j for j in old['inputs']['jobs']if j['id']==p['failed_job_id'])
 if job['kind']!='prefix':raise ValueError('PREFIX only')
 original=run.io.read_json(root,p['failed_result'])
 if original.get('object_id')!=job['object_id']or original['models']['linear_rest_tangent']['files']!=p['failed_model_files']:raise ValueError('Exact failed model source binding')
 for r in p['failed_model_files'].values():run.io.checked(root,r)
 run.io.local(root,p['output']);return p,old,job


def execute(protocol,digest,*,root=ROOT):
 root=Path(root).resolve();p,old,j=validate(protocol,digest,root=root);run.io.guard(root,RESOURCES['CPU_cores']);out=run.io.local(root,p['output'])
 if out.exists():raise FileExistsError('No trace overwrite/retry')
 out.mkdir(parents=True);start=time.monotonic();trace=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),query_count=0,last_queries=[],offending_queries=[],v02_guard_calls=True,private_truth_access=False)
 def save():trace['seconds']=time.monotonic()-start;run.io.atomic(out/'summary.json',trace)
 def timeout(*args):raise TimeoutError('Public one-case CDFtrace120s cap')
 previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(120);save()
 try:
  design=run.oldside.obs.load_frozen_design(root);public=run.oldside.prefix_codec.load_public_prefix(root,j['prefix'],design=design)
  # The exact already-derived initial linear library only: no SIDE/DOWN Ys,
  # nonlinear response files or private target/parameter members are loaded.
  model=run.io.load_linear_library(root,old['inputs']['model_inputs']['linear_library']);models,targets=prefix.compile_public_models(model,public,design=design);post=prefix.PolynomialPosterior(models,method='GL128',order=128)
  for E in np.geomspace(75000.,150000.,17):post.cdf(E)
  for probability in prefix.PROBABILITIES:prefix.material_quantile(post,probability)
  post.target_moments(targets,scale_m=200e-6)
  for point in prefix.POINT_IDS:
   for axis in range(3):
    coef={c:v[:,int(point)*3+axis]for c,v in targets.items()}
    for probability in prefix.PROBABILITIES:
     run.io.guard(root,RESOURCES['CPU_cores']);trace['coordinate']=dict(point_id=int(point),axis=axis,probability=probability);trace_coordinate(post,coef,probability,trace)
  trace.update(completed=True,old_guard_failure_reproduced=False)
 except Exception as error:
  trace.update(completed=True,old_guard_failure_reproduced=isinstance(error,FloatingPointError)and 'Finite ordered numerical CDF enclosure required'in str(error),failure=dict(type=type(error).__name__,message=str(error)))
  if isinstance(error,TimeoutError):trace['completed']=False
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous);save()
 return trace


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a.add_argument('--resolved-protocol');a.add_argument('--resolved-protocol-sha256');a.add_argument('--failed-job-id');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output,resolved_protocol=v.resolved_protocol,resolved_protocol_sha256=v.resolved_protocol_sha256,failed_job_id=v.failed_job_id))
 else:print(json.dumps({'completed':execute(v.protocol,v.protocol_sha256)['completed']}))
