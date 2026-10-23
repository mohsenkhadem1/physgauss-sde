"""Manufactured two-candidate worker cost; no actual archives/prefixes/draws."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_c38_public_prefix_inference_v02 import infer_public_prefix
from experiments.campaign_c38_candidate_point_forward import CandidatePointForward
from experiments.campaign_c38_linear_candidate_forward import LinearCandidatePointForward
from experiments.campaign_c38_controlled_observations import public_prefix_payload,POINT_IDS
from experiments.campaign_c38_nonlinear_table_catalog import catalog
from experiments.campaign_c38_table_validation import POINTS
from experiments.run_campaign_c38_static_prequalification import cases
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c38-public-prefix-inference-v02'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def design():
 return dict(schema='C38-controlled-static-ideal-inverse-DESIGN-v01',sensor=dict(kind='IDEAL_TRACKED_MATERIAL_POINTS_NOT_RGBD',qualified_point_ids=POINT_IDS.tolist(),reference_points_m=POINTS[POINT_IDS].tolist(),noise=dict(distribution='independent_zero_mean_Gaussian_per_Cartesian_coordinate_per_acquisition',sigma_m=.0002,known_stipulated_not_sensor_calibrated=True)),acquisition=dict(prefix_Fz_N=0.,candidate_probe_Fz_N=[-2.,2.],target_Fz_N=1.,same_latent_across_resets=True,each_probe_reset_from_common_rest=True),noise_repeats_per_configuration=16,independent_noise_seeds=dict(root_seed=2026102301))
def coefficients(support,F,nonlinear):
 k=1. if support=='both_ends' else 1.12;X=POINTS
 g=np.stack([.0003*(1+X[:,0]/.24),.0002*(1+.2*X[:,1]/.04),-.001*(1+.1*np.cos(X[:,0]/.24))],1);c=np.zeros((3,63,3));c[1]=k*g*(1+.12*F)
 if nonlinear:c[2]=k*g*.15*(1+.08*F)
 return c

def manufactured_inputs():
 rows=[]
 for index,c in enumerate(catalog()):
  if c['material_use']=='TABLE_NODE':
   for j,nx in enumerate((32,40)):rows.append((2*index+j,c,nx))
 q=[]
 for _,c,nx in rows:
  a=coefficients(c['support'],c['Fz_N'],True);x=75000./c['young_Pa'];q.append(POINTS+a[0]+x*(a[1]+x*a[2]))
 archive=dict(source_index=np.array([r[0] for r in rows]),case_ids=np.array([r[1]['id'] for r in rows]),material_use=np.array(['TABLE_NODE']*48),information_role=np.array('CANDIDATE_POINT_FORWARD_TABLE_NOT_ADMITTED'),cells=np.array([[n,n//2,n//2] for _,c,n in rows]),support=np.array([r[1]['support'] for r in rows]),young_Pa=np.array([r[1]['young_Pa'] for r in rows]),Fz_N=np.array([r[1]['Fz_N'] for r in rows]),material_probes_m=POINTS.copy(),probe_positions_m=np.array(q));linear=[]
 for c in cases():
  a=coefficients(c['support'],c['Fz_N'],False);x=75000./c['young_Pa']
  for nx in (32,40):linear.append(dict(case=c,cells=[nx,nx//2,nx//2],material_probes_m=POINTS.copy(),probe_positions_m=POINTS+a[1]*x))
 a=coefficients('both_ends',0.,True);q=POINTS+a[1]*.7+a[2]*.7**2;fake_error=.05*.0002*np.sin(np.arange(27)).reshape(9,3);payload=public_prefix_payload('a'*32,q[POINT_IDS]+fake_error,design=design())
 return {'nonlinear':CandidatePointForward(archive),'linear':LinearCandidatePointForward(linear)},payload

def prepare():
 if (OP/'manufactured-protocol.json').exists():raise FileExistsError('Preserveprotocol')
 files=['experiments/campaign_c38_public_prefix_inference_v02.py','tests/test_campaign_c38_public_prefix_inference_v02.py','experiments/benchmark_c38_public_prefix_inference_v02.py','experiments/campaign_c38_candidate_point_forward.py','experiments/campaign_c38_linear_candidate_forward.py','experiments/campaign_c38_controlled_observations.py','experiments/campaign_c38_polynomial_likelihood_v02.py','experiments/campaign_c38_polynomial_cache.py']
 p=dict(schema='manufactured-public-prefix-worker-runtime-v2',superseded_root_resolution='v01preserved1e-10m/1e-10unit; v02prospective1e-8m/1e-7unit BEFOREactualobservations; allcomparisongatesunchanged',preserved_v01_result=dict(path='runs/operator-dev/c38-public-prefix-inference-v01/manufactured-result.json',sha256=sha(ROOT/'runs/operator-dev/c38-public-prefix-inference-v01/manufactured-result.json')),status='FROZEN_BEFORE_SYNTHETIC_WORKER',source_sha256={f:sha(ROOT/f) for f in files},CPU_affinity_cores=[30,31],hard_per_candidate_s=300,candidates=['nonlinear','linear'],data='All48nonlinear/28linear candidate fields manufactured from explicitknownpolynomials; SAME synthetic publicprefix; deterministicfakeerror, noactualdraw orarchive/prefix reads',scope='All63moments,27coordinate95/midquantiles,3numericalmethods,GL64checks. NumericalSOURCEcontrol only; no physicaladmission/calibration',future_controlled_execution='NO LAUNCH AUTHORIZATION; rootowns future8CPU2workers6..21/RSS2GiBeach/global3600protocol')
 write(OP/'manufactured-protocol.json',p);return sha(OP/'manufactured-protocol.json')
class Cap(BaseException):pass

def run(digest):
 p=json.loads((OP/'manufactured-protocol.json').read_text())
 if sha(ROOT/p['preserved_v01_result']['path'])!=p['preserved_v01_result']['sha256'] or sha(OP/'manufactured-protocol.json')!=digest or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedpins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!={30,31}:raise ValueError('CPU2only')
 models,payload=manufactured_inputs();s=dict(completed=False,protocol_sha256=digest,candidates={},actual_data_access=False)
 for name in p['candidates']:
  t=time.monotonic();signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(Cap('Manufactured300scandidatecap')));signal.alarm(300)
  try:
   result=infer_public_prefix(models[name],payload,design=design());summary=dict(completed=result['completed'],numerical_controls_passed=result['numerical_controls_passed'],failures=result['failures'],comparisons=result.get('comparisons'),maximum_GL64_endpoint_CDF_difference=max([r['CDF_difference'] for r in result.get('GL64_endpoint_CDF_checks',[])] or [0.]),cache_stats=result.get('cache_stats'),method_completion={m:dict(completed=r['completed'],stage=r['stage'],error=r.get('error')) for m,r in result['methods'].items()})
  except BaseException as error:summary=dict(completed=False,numerical_controls_passed=False,error=repr(error))
  finally:signal.alarm(0)
  summary['elapsed_s']=time.monotonic()-t;s['candidates'][name]=summary;write(OP/'manufactured-result.json',s)
 s['completed']=all(r['completed'] for r in s['candidates'].values());s['all_numerical_controls_passed']=all(r['numerical_controls_passed'] for r in s['candidates'].values());write(OP/'manufactured-result.json',s);return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--sha');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.sha);print(json.dumps(r if isinstance(r,str) else dict(completed=r['completed'],all_numerical_controls_passed=r['all_numerical_controls_passed'],times={k:v['elapsed_s'] for k,v in r['candidates'].items()})))
