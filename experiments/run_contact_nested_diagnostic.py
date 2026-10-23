"""Prospective saved-field nested-contact diagnostic; no trajectory API."""
from pathlib import Path
import argparse,hashlib,json,os,signal,time
import numpy as np
from scipy import sparse
from experiments import campaign_contact_nested_mesh as control
ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/run_contact_nested_diagnostic.py'
RESOURCES=dict(CPU_cores=[6,7],threads=2,no_CUDA=True,wall_s=180,RSS_bytes=2*1024**3)
BASE='runs/operator-dev/p2-corrected-contact-order128-trajectory-v01/protocol.json'
SUMMARY='runs/gaussian-campaign-v01/p2-corrected-contact-order128-trajectory-v01/outer128_accuracy_1e10/summary.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return dict(path=str(path),sha256=sha(ROOT/path))
def write(path,data):
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def checked(ref):
 path=ROOT/ref['path']
 if sha(path)!=ref['sha256']:raise ValueError('Changed frozen diagnostic input: '+ref['path'])
 return path


def sources():
 base=json.loads((ROOT/BASE).read_text())
 return sorted(set(base['source_sha256'])|{SOURCE,'experiments/campaign_contact_nested_mesh.py','tests/test_campaign_contact_nested_mesh.py','tests/test_run_contact_nested_diagnostic.py','experiments/campaign_p2_contact_adaptive_energy_gradient.py','experiments/campaign_p2_contact_root_fixed.py','experiments/campaign_p2_contact_root_integration.py','experiments/campaign_p2_consistent_transient.py','experiments/campaign_stable_transient.py'})


def prepare(protocol,output):
 if (ROOT/protocol).exists()or(ROOT/output).exists():raise FileExistsError('New exclusive source/diagnostic namespace required')
 base=json.loads((ROOT/BASE).read_text());summary=json.loads((ROOT/SUMMARY).read_text());capture=summary['capture'];checked(capture)
 if not summary.get('completed')or not summary.get('passed'):raise ValueError('Completed immutable original-kappa coarse capture required')
 clocks=base['outputs_s']
 if len(clocks)!=9:raise ValueError('Exactly existing nine clocks required')
 p=dict(schema='CONTACT-nested96to768-readonly-v01',status='SOURCE_ONLY_FROZEN_BEFORE_SAVED_FIELD_CALLS',output=output,sources={n:sha(ROOT/n)for n in sources()},runtime=dict(numpy=np.__version__),resources=RESOURCES,inputs=dict(original_protocol=record(BASE),coarse_summary=record(SUMMARY),coarse_capture=capture),coarse_cells=[4,2,2],fine_cells=[8,4,4],clocks_s=clocks,physics=dict(E_Pa=300000.,nu=.35,rho_kg_m3=625.,gravity_m_s2=9.81,kappa_N_m3=1e9,friction=0.,damping=0.,plane_z_m=0.,law='Unchanged whole-boundary reference-area P2 compliant-plane energy'),checks=['Actual dyadic tetrahedral containment and coarse face partition before data access','FP64 physical-coordinate reconstruction budget, not exact-real claim for decimal coordinates','P2 constants/affines/quadratics; sparse signed consistent-mass restriction','Identical coarse field prolonged to768: root128 U and dual restricted gradient','Independent strict adaptive E/g, all faces successful; no status waiver/partial reference'],gates=dict(energy_J=1e-12,gradient_L2_N=1e-10,adaptive_energy_atol_J=1e-12,adaptive_gradient_atol_N=1e-10,adaptive_rtol=1e-10),no_physics_or_trajectory_calls=True,no_dataset_admission=True,raw_saved_fields_local=True)
 write(ROOT/protocol,p);return sha(ROOT/protocol)


def validate(protocol,digest):
 if sha(ROOT/protocol)!=digest:raise ValueError('Changed recipe')
 p=json.loads((ROOT/protocol).read_text())
 if p.get('schema')!='CONTACT-nested96to768-readonly-v01'or p.get('resources')!=RESOURCES or p.get('sources')!={n:sha(ROOT/n)for n in sources()}or p.get('runtime')!=dict(numpy=np.__version__):raise ValueError('Frozen source/resource contract')
 for r in p['inputs'].values():checked(r)
 return p


def guard():
 if set(os.sched_getaffinity(0))!={6,7}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='2'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact CPU6,7/thread2/noCUDA contract')
 rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
 if rss>RESOURCES['RSS_bytes']:raise MemoryError('RSS2GiB cap')


def run(protocol,digest):
 p=validate(protocol,digest);guard();out=ROOT/p['output']
 if out.exists():raise FileExistsError('No retry or overwrite')
 out.mkdir(parents=True);start=time.monotonic();result=dict(completed=False,passed=False,pid=os.getpid(),protocol=dict(path=protocol,sha256=digest),states=[],new_trajectory_calls=0,raw_fields_written=False)
 def save():
  result['seconds']=time.monotonic()-start;(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 def timeout(*a):raise TimeoutError('Nested saved-field CPU180s cap')
 previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
 try:
  coarse,fine=control.geometry(tuple(p['coarse_cells'])),control.geometry(tuple(p['fine_cells']));P,nesting=control.nested_transfer(coarse,fine);result.update(nesting=nesting,polynomial_controls=control.polynomial_checks(coarse,fine,P),signed_mass=control.mass_check(coarse,fine,P));save()
  # Geometry/control proof precedes access to actual saved deformations.
  with np.load(checked(p['inputs']['coarse_capture']),allow_pickle=False)as z:
   for key,value in [('rest_m',coarse.rest),('tets',coarse.tets),('elements',coarse.elements),('faces',coarse.faces),('times_s',np.array(p['clocks_s']))]:np.testing.assert_array_equal(z[key],value)
   q=z['positions_m'].copy()
  if q.shape!=(9,len(coarse.rest),3):raise ValueError('Exact predeclared nine coarse states')
  for clock,state in zip(p['clocks_s'],q):
   guard();row=dict(clock_s=clock,completed=False,passed=False)
   try:
    refined=P@state;a=control.contact(state,coarse);b=control.contact(refined,fine);ar=control.contact(state,coarse,adaptive=True);br=control.contact(refined,fine,adaptive=True)
    row.update(root128_invariance=control.compare(a,b,P),adaptive_invariance=control.compare(ar,br,P),coarse_root128_vs_adaptive=control.compare(a,ar,sparse.eye(len(coarse.rest),format='csr')),fine_root128_vs_adaptive=control.compare(b,br,sparse.eye(len(fine.rest),format='csr')),adaptive_estimated_energy_errors_J=[ar['adaptive_estimated_E_error_J'],br['adaptive_estimated_E_error_J']],adaptive_sum_local_gradient_error_estimates_N=[ar['adaptive_sum_local_g_error_estimates_N'],br['adaptive_sum_local_g_error_estimates_N']],completed=True)
    row['passed']=all(v['passed']for v in row.values()if isinstance(v,dict)and 'passed'in v)
   except(RuntimeError,ValueError)as e:row['error']=repr(e)
   result['states'].append(row);save()
  result.update(completed=True,passed=all(r['completed']and r['passed']for r in result['states']))
 except Exception as e:result['error']=repr(e)
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous);save()
 return result


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');v=a.parse_args()
 if v.mode=='prepare':print(prepare(v.protocol,v.output))
 else:print(json.dumps(dict(completed=run(v.protocol,v.protocol_sha256)['completed'])))
