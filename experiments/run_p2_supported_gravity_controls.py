"""Small independent sparse/streamed CPU controls, not dataset admission."""
import argparse,hashlib,json,os,signal,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OP=ROOT/'runs/operator-dev/p2-supported-gravity-v01'
OUT=ROOT/'runs/gaussian-campaign-v01/p2-supported-gravity-v01'
SOURCES=['experiments/run_p2_supported_gravity_controls.py',
 'experiments/campaign_p2_supported_preconditioner.py','tests/test_campaign_p2_supported_preconditioner.py',
 'experiments/campaign_p2_geometry.py','experiments/campaign_p2_streaming.py',
 'experiments/campaign_p2_coarse_preconditioner.py','experiments/campaign_p2_block_preconditioner.py',
 'experiments/campaign_p2_torch.py','experiments/campaign_p2_consistent_transient.py',
 'experiments/campaign_p2_jacobian_certificate.py','experiments/campaign_block_surface_controls.py',
 'experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py',
 'experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py',
 'experiments/boundary_block.py','experiments/diagnose_nonlinear_physical_probes.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preserve prior protocol')
 p=dict(schema='C38-supported-gravity-controls-v01',status='FROZEN_BEFORE_CONTROL_SOLVES',source_sha256={n:sha(ROOT/n) for n in SOURCES},cells=[4,2,2],supports=['right_face','both_ends','bottom_face','separated_bottom_patches'],material=dict(E_Pa=100000.,nu=.35,rho_kg_m3=625.),gravity_m_s2=9.81,known_surface_pattern='transverse_end',constant_reference_force_N=[0,0,-.12],solvers=['Independent NumPy/SciPy sparse full Newton','Torch FP64 streamed P2/support-matched coarse preconditioner on CPU'],original_force_tolerance_N=1e-10,allnode_parity_tolerance_m=1e-8,CPU_threads=2,wall_cap_s=300,limits=['96 tetrahedra implementation parity ONLY, not spatially resolved truth','Same original stable polynomial NH/full consistent body loading','No GPU allocations and no learned model or cohort admission','Hard support patches remain as geometrically declared; no support inference'])
 write(OP/'protocol.json',p);print(sha(OP/'protocol.json'))
def run(digest):
 if sha(OP/'protocol.json')!=digest:raise ValueError('Changed protocol')
 p=json.loads((OP/'protocol.json').read_text())
 for n,h in p['source_sha256'].items():
  if sha(ROOT/n)!=h:raise ValueError('Changed source '+n)
 for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
  if os.environ.get(name)!='2':raise ValueError('CPU2 required')
 if OUT.exists():raise FileExistsError('Preserve prior controls')
 OUT.mkdir(parents=True);start=time.monotonic();s=dict(completed=False,passed=False,protocol_sha256=digest,cases=[],dataset_admission=False)
 def expired(*_):raise TimeoutError('Frozen300s CPU controls budget')
 old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  import torch,resource,shutil
  from experiments.campaign_p2_geometry import P2Geometry
  from experiments.campaign_p2_supported_preconditioner import StreamingSupportedP2
  from experiments.campaign_newton_stable_material import StableP2Block
  from experiments.campaign_p2_jacobian_certificate import whole_p2_jacobian_bounds
  torch.set_num_threads(2)
  for kind in p['supports']:
   begin=time.monotonic();m=P2Geometry((4,2,2),young=100000.,gravity=9.81,support=kind)
   cpu=StableP2Block((4,2,2),young=100000.,poisson=.35,density=625.,gravity=9.81,traction=0.,order=3)
   np.testing.assert_array_equal(cpu.rest,m.rest);np.testing.assert_array_equal(cpu.elements,m.elements)
   # Both solvers receive exactly the same declared public dead traction and
   # consistent gravity load. Only the sparse CPU constraint declaration is set.
   cpu.load=m.load.copy();cpu.mask=m.mask.copy();cpu.free=np.flatnonzero(~cpu.mask.ravel())
   a=cpu.solve(force_tolerance_N=1e-10);t=StreamingSupportedP2(m,support_kind=kind);b=t.solve(force_tolerance=1e-10)
   q=b['positions'].numpy();R=(b['cache']['gradient']-t.force).numpy();diag=m.diagnostics(dict(positions=q,residual=R,J=b['cache']['minimum_J'].numpy()[None],tolerance_N=1e-10,iterations=b['iterations']));cert=whole_p2_jacobian_bounds(m,q)
   delta=float(np.linalg.norm(q-a['positions'],axis=1).max());cpuR=float(np.linalg.norm(a['residual'][~m.mask]));passed=bool(delta<=1e-8 and cpuR<=1e-10 and diag['original_free_residual_N']<=1e-10 and diag['force_balance_defect_N']<=1e-8 and diag['actual_moment_balance_defect_Nm']<=1e-8 and diag['maximum_constraint_error_m']==0 and cert['all_elements_positive'])
   f=OUT/(kind+'.npz');np.savez_compressed(f,rest_m=m.rest,elements=m.elements,positions_sparse_CPU_m=a['positions'],positions_streamed_CPU_m=q,constraint_mask=m.mask,known_load_N=m.load)
   row=dict(support=kind,passed=passed,allnode_difference_m=delta,independent_sparse_free_residual_N=cpuR,diagnostics=diag,whole_J_Bernstein_lower=cert['minimum_lower_bound'],sparse_iterations=a['iterations'],streamed_iterations=b['iterations'],elapsed_s=time.monotonic()-begin,capture=dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
   s['cases'].append(row);write(OUT/'summary.json',s)
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>12*1024**3 or shutil.disk_usage(ROOT).free<30*1024**3:raise MemoryError('CPU12GiB/disk30GiB resource guard')
  s.update(completed=True,passed=all(r['passed'] for r in s['cases']))
 except BaseException as e:s.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);s['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',s)
 print(json.dumps({k:s.get(k) for k in ['completed','passed','elapsed_s','error']}))
 return s
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--sha');a=a.parse_args()
 if a.action=='prepare':prepare()
 else:raise SystemExit(0 if run(a.sha)['passed'] else 1)
