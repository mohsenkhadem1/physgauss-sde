"""Four bounded same-discretization stableNH gravity equilibria; not spatial QA."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_newton_stable_material import StableP1Block
from experiments.run_static_supported_block import probes
from experiments.diagnose_nonlinear_physical_probes import interpolate
from experiments.run_static_p2_axial_continuation import sha,dump,record

def prepare(path):
 src=['experiments/run_newton_gravity_reference.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/run_static_supported_block.py','experiments/diagnose_nonlinear_physical_probes.py','experiments/boundary_block.py','experiments/run_static_p2_axial_continuation.py']
 p={'schema':'C38-stable-P1-gravity-protocol-v0.1','status':'FROZEN_BEFORE_GRAVITY_EQUILIBRIA','CPU_threads':2,'hard_wall_cap_s':60,'jobs':[{'young_Pa':E,'cells':list(c)} for E in [300000.,600000.] for c in [(4,2,2),(8,4,4)]],'physics':{'size_m':[.24,.08,.08],'rho':625.,'nu':.35,'gravity_world_m_s2':[0,0,-9.81],'traction_N':0.,'boundary':'Fullleftfaceall3axes','initial':'REST','law':'NewtonstableNHkappa=lambdaL+muL'},'gates':{'original_free_R_N':1e-11,'minimum_J':.1,'exact_clamp_m':0.,'global_force_N':1e-8,'actual_moment_Nm':1e-8},'query_groups':{k:v.tolist() for k,v in probes().items()},'source_sha256':{f:sha(f) for f in src},'qualification':'ExactP1discreteequilibriumreferencesfornativecomparisonONLY;nospatial/dynamic/contact/trainingqualification'}
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 dump(path,p)
def run(protocol,digest,output):
 if sha(protocol)!=digest:raise ValueError('Changedprotocol')
 p=json.loads(Path(protocol).read_text())
 for f,h in p['source_sha256'].items():
  if sha(f)!=h:raise ValueError('Changedsource')
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();points=np.concatenate([np.array(x) for x in p['query_groups'].values()]);d={'schema':'C38-P1-gravity-results-v0.1','completed':False,'protocol':record(protocol),'cases':[],'spatial_or_native_or_contact_qualified':False}
 def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
 signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Frozen60sCPU2gravitybudget')));signal.alarm(60)
 try:
  for job in p['jobs']:
   cells=tuple(job['cells']);E=job['young_Pa'];row={'young_Pa':E,'cells':list(cells),'tetrahedra':int(6*np.prod(cells)),'completed':False};begin=time.monotonic();label=f'E{int(E)}-tets{row["tetrahedra"]}'
   try:
    m=StableP1Block(cells,young=E,poisson=.35,density=625.,gravity=9.81,traction=0.,boundary='left_face');a=m.solve();row.update(m.diagnostics(a));row['completed']=True;row['history']=a['history'];q=interpolate(m.rest,m.tets,a['positions'][None],points)[0][0];U,g,H,J=m.elastic(a['positions']);path=out/(label+'.npz');np.savez_compressed(path,rest_m=m.rest,tets=m.tets,faces=m.faces,positions_m=a['positions'],elastic_forces_N=-g,applied_forces_N=m.load,physical_masses_kg=m.node_mass,constraint_mask=m.mask,material_probes_m=points,probe_positions_m=q);row['capture']=record(path);row['physics_passed']=row['minimum_J']>.1 and row['maximum_constraint_error_m']==0 and row['original_free_residual_N']<=1e-11 and row['force_balance_defect_N']<=1e-8 and row['actual_moment_balance_defect_Nm']<=1e-8
   except TimeoutError as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};raise
   except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])}
   finally:row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/(label+'.json'),row);save()
  d['completed']=True;d['all_equilibria_passed']=all(r.get('physics_passed',False) for r in d['cases'])
 except Exception as e:d['execution_failure']=repr(e)
 finally:signal.alarm(0);save()
 return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
 if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
 else:
  d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','all_equilibria_passed','execution_failure']}));raise SystemExit(0 if d['completed'] else 1)
