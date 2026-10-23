"""Two manufactured complete SIDE54 model executions; no archives/cohort reads."""
from pathlib import Path
import argparse,json,os,resource,signal,time,hashlib
import numpy as np
from experiments import campaign_c39_postprobe_inference as infer
from experiments import campaign_c39_side_observations as codec
from experiments.campaign_c38_table_validation import POINTS
from experiments.campaign_c38_controlled_observations import POINT_IDS,public_prefix_payload
from experiments import run_campaign_c38_initial_inference_v03 as io
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/benchmark_campaign_c39_typed_bundle.py'


def design():
 return dict(schema='C38-controlled-static-ideal-inverse-DESIGN-v01',sensor=dict(kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',qualified_point_ids=POINT_IDS.tolist(),reference_points_m=POINTS[POINT_IDS].tolist(),noise=dict(distribution='independent_zero_mean_Gaussian_per_Cartesian_coordinate_per_acquisition',sigma_m=.0002,known_stipulated_not_sensor_calibrated=True)),acquisition=dict(prefix_Fz_N=0.,candidate_probe_Fz_N=[-2.,2.],target_Fz_N=1.,same_latent_across_resets=True,each_probe_reset_from_common_rest=True),noise_repeats_per_configuration=16,independent_noise_seeds=dict(root_seed=2026102301))


class ManufacturedComposite:
 categories=infer.CATEGORIES;young_nodes=np.array([75000.,np.sqrt(75000.*150000.),150000.])
 def __init__(self,nonlinear):self.nonlinear=nonlinear;self.initial_candidate=self;self.calls=[]
 def observation_coefficients(self,category,action_id,ids):
  self.calls.append((category,action_id,tuple(ids)));coef=np.zeros((3,len(ids),3));factor=1. if category==self.categories[0]else 1.15
  if action_id=='canonical_up_1N':
   for j,p in enumerate(ids):
    if p==POINT_IDS[0]:coef[1,j,2]=.001*factor;coef[2,j,2]=.00005*self.nonlinear
  elif action_id in ('gravity','middle_side_2N'):
   axis=2 if action_id=='gravity'else 1;coef[1,:,axis]=.0001*factor;coef[2,:,axis]=.00001*self.nonlinear
  else:raise AssertionError('NoDOWN observations or scalarFz alias')
  return coef.reshape(3,-1)


def prepare(protocol,output,*,root=ROOT):
 root=Path(root);p=io.local(root,protocol)
 if p.exists()or io.local(root,output).exists():raise FileExistsError('Fresh benchmarkonly')
 names=[SOURCE,'experiments/campaign_c39_postprobe_inference.py','experiments/campaign_c39_side_observations.py','experiments/campaign_c39_side_point_forward.py','experiments/campaign_c38_exact_posterior_memo.py','experiments/campaign_c38_polynomial_likelihood_v02.py','experiments/campaign_c38_polynomial_cache.py']
 p.parent.mkdir(parents=True,exist_ok=True);io.write_new(p,dict(schema='C39-manufactured-typed-bundle-benchmark-v01',source_sha256={n:io.sha(root/n)for n in names},output=output,resources=dict(cores=[30,31],threads=2,global_wall_s=180,RSS_bytes=2*1024**3,no_CUDA=True),runtime=io.runtime(),actual_data_reads=False,models=['manufactured_nonlinear','manufactured_linear'],all_three_methods_required=True));return io.sha(p)


def run(protocol,digest,*,root=ROOT):
 root=Path(root);p=io.read_json(root,dict(path=protocol,sha256=digest));assert p['runtime']==io.runtime()
 for n,h in p['source_sha256'].items():io.checked(root,dict(path=n,sha256=h))
 if set(os.sched_getaffinity(0))!={30,31}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU2/noGPU')
 out=io.local(root,p['output']);out.mkdir(parents=True,exist_ok=False);state=dict(completed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),models={},actual_data_reads=False);start=time.monotonic()
 def cap(*a):raise TimeoutError('Manufactured180s cap')
 old=signal.signal(signal.SIGALRM,cap);signal.alarm(180)
 np.load=lambda*a,**k:(_ for _ in()).throw(AssertionError('Actual arrayreads forbidden'))
 np.random.default_rng=lambda*a,**k:(_ for _ in()).throw(AssertionError('No new noise draw'))
 try:
  public=public_prefix_payload('a'*32,POINTS[POINT_IDS].copy(),design=design());one=codec.public_probe(public,'middle_side_2N',POINTS[POINT_IDS].copy(),design=design())
  for nonlinear,name in [(True,'manufactured_nonlinear'),(False,'manufactured_linear')]:
   model=ManufacturedComposite(nonlinear);t=time.monotonic();r=infer.infer_public_probe(model,public,one,design=design());io.write_new(out/(name+'.json'),io.jsonable(r));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
   if rss>p['resources']['RSS_bytes']:raise MemoryError('2GiB benchmarkcap')
   state['models'][name]=dict(completed=r['completed'],numerical_controls_passed=r['numerical_controls_passed'],seconds=time.monotonic()-t,RSS_peak_bytes=rss,all_methods={n:dict(completed=x['completed'],stage=x['stage'])for n,x in r['methods'].items()},comparisons=r.get('comparisons'),maximum_endpoint_CDF_difference=max([c['CDF_difference']for c in r.get('GL64_endpoint_CDF_checks',[])]or[0]),result=io.record(root,out/(name+'.json')),typed_actions_called=sorted({a for c,a,ids in model.calls}),failures=r['failures']);io.atomic(out/'summary.json',state)
  state['completed']=all(x['completed']and x['numerical_controls_passed']for x in state['models'].values())
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);state['seconds']=time.monotonic()-start;io.atomic(out/'summary.json',state)
 return state


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 else:r=run(v.protocol,v.protocol_sha256);print(json.dumps(dict(completed=r['completed'],seconds=r['seconds'])));raise SystemExit(0 if r['completed']else 1)
