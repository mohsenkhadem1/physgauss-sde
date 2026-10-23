"""Finite CPU-only cost/parity probe of synthetic27-observation/189-target model."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from experiments.campaign_c38_polynomial_likelihood_v02 import PolynomialPosterior
from experiments.campaign_c38_polynomial_cache import CachedGaussianPolynomialLikelihood as GaussianPolynomialLikelihood
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c38-polynomial-cache-v01'
UNCACHED=ROOT/'runs/operator-dev/c38-polynomial-likelihood-v02/manufactured-result.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def prepare():
 if (OP/'manufactured-protocol.json').exists():raise FileExistsError('Preserveprotocol')
 files=['experiments/benchmark_c38_polynomial_cached.py','experiments/campaign_c38_polynomial_cache.py','tests/test_campaign_c38_polynomial_cache.py','experiments/benchmark_c38_polynomial_manufactured.py','experiments/campaign_c38_polynomial_likelihood_v02.py','tests/test_campaign_c38_polynomial_likelihood_v02.py']
 p=dict(schema='exact-cached-manufactured-polynomial-runtime-v1',reference_uncached_result=dict(path=str(UNCACHED.relative_to(ROOT)),sha256=sha(UNCACHED)),status='FROZEN_BEFORE_SYNTHETIC_PROBE',source_sha256={f:sha(ROOT/f) for f in files},CPU_affinity_cores=[30,31],hard_cap_s=120,observation_dimensions=27,target_dimensions=189,noise_sigma_m=200e-6,acquired_observed='Deterministic syntheticquadratic meanatx=.7 plus .05sigma*cos(index); NOphysicaltruth/arrays',numerics=['adaptive','legendre128'],material_CDF_queries=17,material_quantiles=[.025,.5,.975],canonical_marginal_interval_scope='ONE syntheticcoordinate, threequantiles perrule. NOTall27physicalintervalcoordinates.',targets='Separate syntheticcoefficients ofnoiselesslatentfield, notobservations',actual_cohort_or_physics_access=False)
 write(OP/'manufactured-protocol.json',p);return sha(OP/'manufactured-protocol.json')
def run(digest):
 p=json.loads((OP/'manufactured-protocol.json').read_text())
 if sha(OP/'manufactured-protocol.json')!=digest or sha(UNCACHED)!=p['reference_uncached_result']['sha256'] or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31}:raise ValueError('CPU2only')
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Synthetic120scap')));signal.alarm(120);begin=time.monotonic();s=dict(completed=False,protocol_sha256=digest,rows=[])
 try:
  sigma=200e-6;i=np.arange(27);c=np.stack([np.zeros(27),np.linspace(.001,.003,27),np.linspace(.0001,.0002,27)]);observed=c[0]+c[1]*.7+c[2]*.7**2+.05*sigma*np.cos(i);C=np.eye(27)*sigma**2
  t=time.monotonic();models={'a':GaussianPolynomialLikelihood(c,observed,C),'b':GaussianPolynomialLikelihood(c*1.04,observed,C)};s['model_construction_s']=time.monotonic()-t
  ctarget=np.stack([np.zeros(189),np.linspace(-.003,.003,189),.0002*np.cos(np.arange(189))]);targets={'a':ctarget,'b':ctarget*1.03};allrows=[]
  for method in ['adaptive','legendre']:
   started=time.monotonic();post=PolynomialPosterior(models,method=method,order=128);row=dict(method=method,posterior_construction_s=time.monotonic()-started,logZ=post.log_normalizer,category_probability=post.category_probabilities.tolist());t=time.monotonic();row['material_CDF']=[post.cdf(E) for E in np.geomspace(75000.,150000.,17)];row['material_quantiles_Pa']=[brentq(lambda E:post.cdf(E)-u,75000.,150000.,xtol=1e-5) for u in [.025,.5,.975]];row['material_queries_s']=time.monotonic()-t
   t=time.monotonic();mom=post.target_moments(targets);row['target189_two_pass_moments_s']=time.monotonic()-t;row['target_mean_m']=mom['mean_m'].tolist();row['target_variance_m2']=mom['variance_m2'].tolist();t=time.monotonic();co={k:v[:,0] for k,v in targets.items()};row['one_coordinate_quantiles_m']=[brentq(lambda v:post.coordinate_cdf(co,v)['cdf']-u,-.01,.01,xtol=1e-11) for u in [.025,.5,.975]];row['one_coordinate_three_quantiles_s']=time.monotonic()-t;row['total_s']=time.monotonic()-started;row['cache_stats']={k:v.cache_stats() for k,v in models.items()};s['rows'].append(row);write(OP/'manufactured-result.json',s)
  a,b=s['rows'];comparison=dict(logZ_difference=abs(a['logZ']-b['logZ']),category_difference=float(max(abs(np.array(a['category_probability'])-b['category_probability']))),CDF_difference=float(max(abs(np.array(a['material_CDF'])-b['material_CDF']))),logE_quantile_difference=float(max(abs(np.log(np.array(a['material_quantiles_Pa'])/b['material_quantiles_Pa'])))),target_mean_difference_m=float(max(abs(np.array(a['target_mean_m'])-b['target_mean_m']))),target_variance_difference_m2=float(max(abs(np.array(a['target_variance_m2'])-b['target_variance_m2']))),one_coordinate_interval_endpoint_difference_m=float(max(abs(np.array(a['one_coordinate_quantiles_m'])-b['one_coordinate_quantiles_m']))));old=json.loads(UNCACHED.read_text());keys=['logZ','category_probability','material_CDF','material_quantiles_Pa','target_mean_m','target_variance_m2','one_coordinate_quantiles_m'];identity=[dict(method=a['method'],all_numeric_functionals_bit_identical=all(np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in keys),uncached_total_s=b['total_s'],cached_total_s=a['total_s'],uncached_to_cached_time_ratio=b['total_s']/a['total_s']) for a,b in zip(s['rows'],old['rows'])];s.update(completed=True,cache_equivalence_passed=all(r['all_numeric_functionals_bit_identical'] for r in identity),cached_to_uncached_identity=identity,comparison=comparison,actual_data_access=False,physical_admission=False)
 except Exception as error:s['error']=repr(error);raise
 finally:signal.alarm(0);s['elapsed_s']=time.monotonic()-begin;write(OP/'manufactured-result.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else {k:r.get(k) for k in ['completed','elapsed_s','error']}))
