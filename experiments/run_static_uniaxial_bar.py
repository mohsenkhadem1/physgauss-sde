"""Prospectively bounded static homogeneous-bar patch benchmark; no inverse."""
from __future__ import annotations
import argparse,hashlib,json,time,signal
from pathlib import Path
import numpy as np
from experiments.campaign_static_uniaxial_bar import PrismaticBar,SIZE,YOUNG,POISSON,SPRING

ROOT=Path(__file__).resolve().parents[1]
MESHES=((2,2,2),(4,4,4))
FORCES=(1.,3.)
CAP=300


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def prepare(path):
 path=Path(path)
 if path.exists():raise ValueError('Fresh static benchmark protocol required')
 sources=['experiments/campaign_static_uniaxial_bar.py','experiments/run_static_uniaxial_bar.py','tests/test_campaign_static_uniaxial_bar.py',
  'experiments/campaign_sparse_nonlinear_gaussian.py','experiments/boundary_block.py','experiments/campaign_nonlinear_excitation.py',
  'experiments/campaign_nonlinear_gaussian_reference.py','experiments/campaign_boundary_gaussians.py',
  'experiments/campaign_compliant_gaussian_linear.py','experiments/campaign_nonlinear_fem.py','experiments/campaign_linear_fem.py','experiments/campaign_material_gaussians.py']
 protocol=dict(schema='static-uniaxial-NH-bar-benchmark-v0.1',status='FROZEN_BEFORE_NONZERO_STATIC_SOLVES',CPU_threads=2,wall_cap_s=CAP,
  physics=dict(young_Pa=YOUNG,poisson=POISSON,size_m=list(SIZE),left_total_axial_spring_N_m=SPRING,right_known_forces_N=list(FORCES),
   load='Uniform nominal deadtraction on reference rightface; consistent triangle nodalarea quadrature',
   fixture='Finite axial springK/A integrated over reference leftface; no lateral stiffness or exact axial pin',
   lateral_faces='Tractionfree',exact_constraints='Localuy=0 on referenceY=0; localuz=0 on referenceZ=0 symmetryplanes',
   static='No mass, velocity, damping or time increment; stationarity of W+Uspring-Fdead.u'),
  meshes=[dict(cells=list(m),tetrahedra=6*int(np.prod(m))) for m in MESHES],
  order='Meshcoarse/F1,F3 then meshfine/F1,F3; restinitialstate eachcell, stop on genuinefailure and preserveunlaunched',
  independent_oracle='Fdiag(s,t,t); y=t^2 solves y-1+(lambda/mu)log(sy)=0; A*mu*(s-y/s)=F; leftshiftF/K',
  numerical=dict(maximum_iterations=40,force_residual_N='max(1e-10,1e-10*totalforce)',Armijo=1e-4,minimum_J=.1,
   stiffness_shift_or_physics_repair=False,mass_or_fake_timestep=False,exact_constraint_elimination=True),
  gates=dict(maximum_position_error_m=2e-8,maximum_firstPiola_error_Pa=2e-5,force_and_spring_balance_error_N=5e-9,
   maximum_energy_error_J=1e-10,maximum_kinematic_symmetry_error_m=0.,axial_strain_range=[.02,.08],coarse_fine_common_node_max_error_m=2e-8),
  tests=dict(passed=6,elapsed_s=1.71,scope='Analyticforce/lateralbalance, zerostaticrest, independentenergygradient/tangent, finiteframecovariance, oracle stress/energy/nodal equilibrium, invalidJ/constraintfail'),
  source_sha256={p:sha(ROOT/p) for p in sources},
  scope='Homogeneous finite-strain static analytic patch verification ONLY; not Gaussianpatchmesh qualification, temporalconvergence, inverse, sensors, Isaac or generalboundary validation',
  no_prior_source_edits_or_cohort_values=True,failures_retained_no_retry=True,raw_arrays_LOCAL=True)
 path.parent.mkdir(parents=True,exist_ok=True);write(path,protocol);return rec(path)


def one_cell(cells,force,output):
 model=PrismaticBar(cells,force=force);start=time.monotonic();answer=model.solve();u=answer['displacement_local_m'];oracle_u,oracle=model.oracle_field()
 F,P=model.independent_stress(answer['positions']);expected_P=np.diag([oracle['first_Piola_axial_Pa'],0,0])
 expected_F=np.diag([oracle['stretch'],oracle['lateral_stretch'],oracle['lateral_stretch']])
 cauchy=np.einsum('eiJ,ekJ->eik',P,F)/np.linalg.det(F)[:,None,None];expected_cauchy=np.diag([oracle['first_Piola_axial_Pa']*oracle['stretch']/oracle['J'],0,0])
 leftspring=float(np.sum(model.Kdiagonal[:,0]*u[:,0]));force_from_stress=float(np.average(P[:,0,0],weights=model.volumes)*SIZE[1]*SIZE[2])
 errors=dict(maximum_position_error_m=float(np.max(abs(u-oracle_u))),coordinate_RMSE_m=float(np.sqrt(np.mean((u-oracle_u)**2))),
  maximum_deformation_gradient_error=float(np.max(abs(F-expected_F))),maximum_firstPiola_error_Pa=float(np.max(abs(P-expected_P))),
  maximum_Cauchy_error_Pa=float(np.max(abs(cauchy-expected_cauchy))),left_spring_force_error_N=abs(leftspring-force),
  stress_integral_force_error_N=abs(force_from_stress-force),elastic_energy_error_J=abs(answer['elastic_J']-oracle['elastic_energy_J']),
  spring_energy_error_J=abs(answer['spring_J']-oracle['spring_energy_J']),external_work_error_J=abs(answer['work_J']-oracle['external_work_J']))
 residual=answer['residual'];rnorm=float(np.linalg.norm(residual.ravel()[model.free]));reaction=float(np.max(abs(residual[model.mask])))
 passed=errors['maximum_position_error_m']<=2e-8 and errors['maximum_firstPiola_error_Pa']<=2e-5 and max(errors['left_spring_force_error_N'],errors['stress_integral_force_error_N'])<=5e-9
 passed=passed and max(errors[k] for k in ('elastic_energy_error_J','spring_energy_error_J','external_work_error_J'))<=1e-10
 passed=passed and np.max(abs(u[model.mask]))==0 and .02<=oracle['stretch']-1<=.08 and rnorm<=answer['tolerance_N']
 label='cells'+''.join(map(str,cells))+'-F'+str(int(force));raw=output/(label+'.npz')
 np.savez_compressed(raw,rest_m=model.rest,tets=model.tets,displacement_m=u,positions_m=answer['positions'],analytic_displacement_m=oracle_u,
  deformation_gradient=F,first_Piola_Pa=P,Cauchy_Pa=cauchy,known_nodal_force_N=model.load,left_area_weights=model.left_area,right_area_weights=model.right_area)
 report=dict(id=label,cells=list(cells),nodes=len(model.rest),tetrahedra=len(model.tets),known_force_N=force,completed=True,passed=bool(passed),
  elapsed_s=time.monotonic()-start,Newton_iterations=answer['iterations'],minimum_J=float(answer['J'].min()),
  original_free_residual_N=rnorm,threshold_N=answer['tolerance_N'],maximum_transverse_symmetry_reaction_N=reaction,
  exact_constraint_max_error_m=float(np.max(abs(u[model.mask]))),oracle=oracle,errors=errors,history=answer['history'],raw=rec(raw))
 write(output/(label+'.json'),report);return report,model,u


def run(protocol_path,digest,output):
 path=ROOT/protocol_path
 if sha(path)!=digest:raise ValueError('Static protocol changed')
 protocol=json.loads(path.read_text())
 for p,h in protocol['source_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('Static benchmark source changed:'+p)
 if protocol['wall_cap_s']!=CAP or protocol['physics']['right_known_forces_N']!=list(FORCES):raise ValueError('Fixed prospective benchmark required')
 output=Path(output)
 if output.exists():raise ValueError('Fresh static output; retain oldfailures')
 output.mkdir(parents=True);write(output/'protocol.json',protocol);start=time.monotonic()
 result=dict(schema='static-uniaxial-NH-bar-result-v0.1',completed=False,passed=False,protocol=dict(path=protocol_path,sha256=digest),
  cells=[],mesh_comparisons=[],unlaunched=[dict(cells=list(m),force_N=f) for m in MESHES for f in FORCES],CPU_threads=2,
  Gaussian_boundary_or_dynamics_qualification=False,inverse_fit=False,raw_arrays_LOCAL=True)
 def save():result['elapsed_s']=time.monotonic()-start;write(output/'summary.json',result)
 def alarm(_s,_f):raise TimeoutError('Declared300s static benchmark cap reached')
 old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,CAP);coarse={}
 try:
  for mesh in MESHES:
   for force in FORCES:
    report,model,u=one_cell(mesh,force,output);result['cells'].append(report);result['unlaunched'].pop(0);save()
    if not report['passed']:raise RuntimeError('Static analytic patch accuracy gate failed; preserve terminalcell')
    if mesh==MESHES[0]:coarse[force]=(model,u)
    else:
     c,cu=coarse[force];lookup={tuple(np.round(x,13)):i for i,x in enumerate(model.reference_local)}
     indices=[lookup[tuple(np.round(x,13))] for x in c.reference_local];delta=float(np.max(abs(cu-u[indices])))
     result['mesh_comparisons'].append(dict(force_N=force,maximum_common_node_displacement_difference_m=delta,passed=delta<=2e-8))
  result.update(completed=True,passed=all(c['passed'] for c in result['cells']) and all(c['passed'] for c in result['mesh_comparisons']),
   interpretation='Analytically representable homogeneous P1 patch accuracy across2meshes; NOT evidence for arbitrarylocalizedGaussianboundary spatialconvergence')
 except Exception as exc:
  result.update(error_type=type(exc).__name__,error=str(exc),failure_history=getattr(exc,'history',[]))
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);save()
 return {k:result[k] for k in ('completed','passed','elapsed_s')}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('prepare');p.add_argument('--output',required=True);p=sub.add_parser('run')
 for name in ('protocol','protocol-sha256','output'):p.add_argument('--'+name,required=True)
 a=parser.parse_args();print(json.dumps(prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output)))
