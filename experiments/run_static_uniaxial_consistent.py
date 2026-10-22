"""Same frozen static oracle benchmark with exact P1 consistent face springs."""
import argparse,json,time,signal
from pathlib import Path
import numpy as np
from experiments import run_static_uniaxial_bar as old
from experiments.campaign_static_uniaxial_consistent import ConsistentPrismaticBar

ROOT=old.ROOT

def prepare(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh consistent-face protocol required')
 parent=ROOT/'runs/operator-dev/static-uniaxial-bar-v01/protocol.json';d=json.loads(parent.read_text())
 d.update(schema='static-consistent-uniaxial-benchmark-v0.2',status='FROZEN_BEFORE_CONSISTENT_STATIC_SOLVES',parent_protocol=old.rec(parent),
  parent_result=old.rec(ROOT/'runs/gaussian-campaign-v01/static-uniaxial-bar-v01/summary.json'),
  variant='Same physics and homogeneous oracle; LEFTspring uses exact triangle integral NiNj=area/12*(ones+I), instead of vertex lumping',
  prior_lumped_patch_result='Valid exact homogeneous patch, preserved unchanged',additional_tests=dict(passed=3,elapsed_s=1.64,scope='Exactaffine-squarefaceintegration, fullenergygradient/tangent, analyticnodalequilibrium'))
 d['source_sha256'].update({p:old.sha(ROOT/p) for p in ['experiments/campaign_static_uniaxial_consistent.py','experiments/run_static_uniaxial_consistent.py','tests/test_campaign_static_uniaxial_consistent.py']})
 p.parent.mkdir(parents=True,exist_ok=True);old.write(p,d);return old.rec(p)


def run(path,digest,output):
 p=ROOT/path
 if old.sha(p)!=digest:raise ValueError('Changed consistent-face protocol')
 d=json.loads(p.read_text())
 for source,h in d['source_sha256'].items():
  if old.sha(ROOT/source)!=h:raise ValueError('Changed frozen source')
 for key in ('parent_protocol','parent_result'):
  if old.sha(ROOT/d[key]['path'])!=d[key]['sha256']:raise ValueError('Changed preserved parent')
 out=Path(output)
 if out.exists():raise ValueError('Fresh output required')
 out.mkdir(parents=True);old.write(out/'protocol.json',d);start=time.monotonic()
 result=dict(schema='static-consistent-uniaxial-result-v0.2',completed=False,passed=False,cells=[],mesh_comparisons=[],unlaunched=[dict(cells=list(m),force_N=f) for m in old.MESHES for f in old.FORCES],
  Gaussian_or_dynamic_mesh_qualification=False,old_lumped_patch_preserved=True,raw_arrays_LOCAL=True)
 def alarm(_s,_f):raise TimeoutError('Declared300s consistentstatic cap reached')
 handler=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,300);coarse={}
 try:
  for cells in old.MESHES:
   for force in old.FORCES:
    model=ConsistentPrismaticBar(cells,force=force);answer=model.solve();u=answer['displacement_local_m'];exact_u,oracle=model.oracle_field();F,P=model.independent_stress(answer['positions'])
    spring_force=(model.spring_matrix@u.ravel()).reshape(u.shape)
    errors=dict(maximum_position_error_m=float(np.max(abs(u-exact_u))),maximum_Piola_error_Pa=float(np.max(abs(P-np.diag([oracle['first_Piola_axial_Pa'],0.,0.])))),
     spring_force_error_N=float(abs(spring_force[:,0].sum()-force)),stress_force_error_N=float(abs(np.average(P[:,0,0],weights=model.volumes)*old.SIZE[1]*old.SIZE[2]-force)),
     elastic_energy_error_J=float(abs(answer['elastic_J']-oracle['elastic_energy_J'])),spring_energy_error_J=float(abs(answer['spring_J']-oracle['spring_energy_J'])),external_work_error_J=float(abs(answer['work_J']-oracle['external_work_J'])))
    residual=float(np.linalg.norm(answer['residual'].ravel()[model.free]));passed=errors['maximum_position_error_m']<=2e-8 and errors['maximum_Piola_error_Pa']<=2e-5 and max(errors['spring_force_error_N'],errors['stress_force_error_N'])<=5e-9
    passed=passed and max(errors[k] for k in ('elastic_energy_error_J','spring_energy_error_J','external_work_error_J'))<=1e-10 and residual<=answer['tolerance_N'] and np.max(abs(u[model.mask]))==0
    label='cells'+''.join(map(str,cells))+'-F'+str(int(force));raw=out/(label+'.npz')
    np.savez_compressed(raw,rest_m=model.rest,tets=model.tets,positions_m=answer['positions'],displacement_m=u,analytic_displacement_m=exact_u,first_Piola_Pa=P,deformation_gradient=F)
    row=dict(id=label,cells=list(cells),tetrahedra=len(model.tets),force_N=force,completed=True,passed=bool(passed),oracle=oracle,errors=errors,
     original_free_residual_N=residual,minimum_J=float(answer['J'].min()),iterations=answer['iterations'],history=answer['history'],raw=old.rec(raw))
    result['cells'].append(row);result['unlaunched'].pop(0)
    if not passed:raise RuntimeError('Consistent-face analytic patch accuracy failed')
    if cells==old.MESHES[0]:coarse[force]=(model,u)
    else:
     m,cu=coarse[force];lookup={tuple(np.round(x,13)):i for i,x in enumerate(model.reference_local)};idx=[lookup[tuple(np.round(x,13))] for x in m.reference_local]
     delta=float(np.max(abs(cu-u[idx])));result['mesh_comparisons'].append(dict(force_N=force,maximum_common_node_error_m=delta,passed=delta<=2e-8))
  result.update(completed=True,passed=all(x['passed'] for x in result['cells']) and all(x['passed'] for x in result['mesh_comparisons']),interpretation='Independent homogeneous static finite-strain patchverification; no arbitraryGaussian/dynamic/continuumqualification')
 except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc),failure_history=getattr(exc,'history',[]))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler);result['elapsed_s']=time.monotonic()-start;old.write(out/'summary.json',result)
 return {k:result[k] for k in ('completed','passed','elapsed_s')}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True);a=sub.add_parser('prepare');a.add_argument('--output',required=True);a=sub.add_parser('run')
 for n in ('protocol','protocol-sha256','output'):a.add_argument('--'+n,required=True)
 a=p.parse_args();print(json.dumps(prepare(a.output) if a.mode=='prepare' else run(a.protocol,a.protocol_sha256,a.output)))
