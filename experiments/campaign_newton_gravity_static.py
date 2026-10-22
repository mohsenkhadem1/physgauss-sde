"""Independent static gravity equilibrium using native VBD proximal relaxation.

Accepted-start/no-contact initialization and zero velocity are algorithmic;
h=1 is pseudo-time. Only original elastic+gravity residual qualifies equilibrium.
"""
from __future__ import annotations
import argparse,importlib.metadata,json,os,subprocess,time,traceback
import numpy as np
from experiments.campaign_newton_contact_controls_v04 import ROOT,ENV,SITE,GPU,sha,write
from experiments.campaign_newton_vbd_probe_v02 import material,actual_input_reference
from experiments.campaign_newton_vbd_static_v02 import assemble,query
from experiments.campaign_newton_static_initialization import accepted_start_solver_class
STAGE='newton-gravity-static-v01';OP=ROOT/'runs/operator-dev'/STAGE;OUT=ROOT/'runs/gaussian-campaign-v01'/STAGE;CPU=ROOT/'runs/gaussian-campaign-v01/newton-gravity-reference-v01'
JOBS=[dict(E=E,tets=n,iterations=512) for E in [300000,600000] for n in [96,768]]+[dict(E=300000,tets=96,iterations=128)]
SOURCES=['experiments/campaign_newton_gravity_static.py','experiments/run_newton_gravity_static.py','tests/test_campaign_newton_gravity_static.py','experiments/campaign_newton_contact_controls_v04.py','experiments/campaign_newton_vbd_probe_v02.py','experiments/campaign_newton_vbd_static_v02.py','experiments/campaign_newton_vbd_static_kernels.py','experiments/campaign_newton_static_initialization.py','experiments/campaign_newton_vbd_kernels.py','experiments/boundary_block.py']
EXTERNAL=['newton/_src/solvers/vbd/solver_vbd.py','newton/_src/solvers/vbd/particle_vbd_kernels.py','newton/_src/sim/builder.py']
CPU_PROTOCOL='runs/operator-dev/newton-gravity-reference-v01/protocol.json'

def gates(R,drift,vmax,J,clamp):
 return bool(np.isfinite([R,drift,vmax,J,clamp]).all() and R<=.01 and drift<=.5e-6 and vmax<=.5e-6 and J>.1 and clamp==0.)

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Frozen protocol exists')
 refs={}
 for j in JOBS:
  label=f"E{j['E']}-tets{j['tets']}";cp=CPU/(label+'.json');d=json.loads(cp.read_text())
  if d.get('completed') is not True:raise ValueError('CPUreference notcompleted')
  for ext in ['.json','.npz']:
   f=CPU/(label+ext);refs[str(f.relative_to(ROOT))]=sha(f)
 p=dict(schema='Newton-static-gravity-proximal-v0.1',status='FROZEN_BEFORE_GPU_CALLS',gpu_uuid=GPU,CPU_threads=2,wall_cap_s=600,jobs=JOBS,source_sha256={f:sha(ROOT/f) for f in SOURCES},external_source_sha256={f:sha(SITE/f) for f in EXTERNAL},CPU_reference_sha256=refs,CPU_protocol=dict(path=CPU_PROTOCOL,sha256=sha(ROOT/CPU_PROTOCOL)),physics=dict(size_m=[.24,.08,.08],rho_kg_m3=625.,nu=.35,gravity_m_s2=[0,0,-9.81],boundary='exactfullleftface',physical_damping=0.,contacts=False),algorithm=dict(h_pseudo_s=1.,zero_velocity_before_each_outer=True,accepted_start_only=True,max_outer=200,fresh_rest_per_job=True),gates=dict(original_native_free_elastic_plus_gravity_R_N=.01,maximum_outer_drift_m=.5e-6,maximum_algorithm_velocity_m_s=.5e-6,minimum_J=.1,fixed_drift_m=0.,maximum_CPU_allnode_position_error_m=20e-6,maximum_CPU_fixedquery_position_error_m=20e-6,maximum_iteration_query_difference_m=20e-6),nonclaims=['Staticnumericalmethod, notphysicaltrajectory','Same-discretizationstableNHcomparison, notcontinuumqualification','FP32restpose/Lame/massforcefloorsreported; CPU1e-11N isnotimposedonnative'])
 write(OP/'protocol.json',p);return dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=sha(OP/'protocol.json'))

def validate(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed protocol')
 p=json.loads((ROOT/path).read_text())
 for key,base in [('source_sha256',ROOT),('external_source_sha256',SITE),('CPU_reference_sha256',ROOT)]:
  if any(sha(base/f)!=v for f,v in p[key].items()):raise ValueError('Changedinput')
 if sha(ROOT/p['CPU_protocol']['path'])!=p['CPU_protocol']['sha256'] or p['jobs']!=JOBS or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Scope/device changed')
 return p

def run(path,digest):
 p=validate(path,digest);OUT.mkdir(parents=True,exist_ok=False);begin=time.monotonic();s=dict(completed=False,passed=False,pid=os.getpid(),protocol=dict(path=path,sha256=digest),cases=[],nonclaims=p['nonclaims'])
 try:
  import warp as wp,newton
  from experiments.campaign_newton_vbd_static_kernels import elastic_local
  versions={k:importlib.metadata.version(k) for k in ['newton','warp-lang','numpy']}
  if versions['newton']!='1.7.0' or versions['warp-lang']!='1.18.0':raise ValueError('Wrongruntime')
  wp.config.kernel_cache_dir=str(OUT/'warp-cache');wp.init();device=wp.get_device('cuda:0');allocation=wp.zeros(1,device=device);wp.synchronize_device(device)
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())];s.update(versions=versions,own_compute_records=own);write(OUT/'summary.json',s)
  if not own or any(r.split(',')[1].strip()!=GPU for r in own):raise ValueError('Wrong physicalGPU')
  fields={}
  for j in JOBS:
   label=f"E{j['E']}-tets{j['tets']}-it{j['iterations']}";start=time.monotonic()
   with np.load(CPU/f"E{j['E']}-tets{j['tets']}.npz",allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
   X=ref['rest_m'];T=ref['tets'];F=ref['faces'];mass=ref['physical_masses_kg'];fixed=np.all(ref['constraint_mask'],axis=1);mu,lam=material(j['E']);builder=newton.ModelBuilder(gravity=(0.,0.,-9.81))
   for i,(x,m) in enumerate(zip(X,mass)):builder.add_particle(tuple(x),(0.,0.,0.),float(m),radius=0.,flags=0 if fixed[i] else int(newton.ParticleFlags.ACTIVE))
   for t in T:builder.add_tetrahedron(*map(int,t),k_mu=mu,k_lambda=lam,k_damp=0.)
   builder.color();model=builder.finalize(device=device);m=model.particle_mass.numpy().astype(float);poses=model.tet_poses.numpy();mats=model.tet_materials.numpy();gravity=np.asarray(model.gravity.numpy(),float).reshape(-1,3)[0]
   np.testing.assert_allclose(gravity,[0,0,-9.81],rtol=1e-7);np.testing.assert_array_equal(model.particle_flags.numpy()[fixed],0);np.testing.assert_allclose(m,mass,rtol=1e-7);np.testing.assert_array_equal(model.tet_materials.numpy(),np.broadcast_to(np.array([mu,lam,0],np.float32),(len(T),3)))
   solver=accepted_start_solver_class(newton.solvers.SolverVBD,wp)(model,iterations=j['iterations'],particle_enable_self_contact=False,particle_enable_tile_solve=False);a=model.state();b=model.state();control=model.control();local=wp.zeros((len(T),4),dtype=wp.vec3,device=device);q0=a.particle_q.numpy().astype(float);q=q0.copy();history=[];force=m[:,None]*gravity
   def elastic(state):
    wp.launch(elastic_local,dim=len(T)*4,inputs=[state.particle_q,model.tet_indices,model.tet_poses,model.tet_materials],outputs=[local],device=device);wp.synchronize_device(device);return assemble(local.numpy(),T,len(X))
   restforce=elastic(a);restfloor=float(np.linalg.norm(restforce[~fixed]))
   for k in range(200):
    a.particle_qd.zero_();a.clear_forces();solver.step(a,b,control,None,1.);wp.synchronize_device(device);newq=b.particle_q.numpy().astype(float);v=b.particle_qd.numpy().astype(float);native=elastic(b);actual=actual_input_reference(newq,T,poses,float(mats[0,0]),float(mats[0,1]));J=actual['J'];R=float(np.linalg.norm((native+force)[~fixed]));drift=float(np.linalg.norm(newq-q,axis=1).max());vmax=float(np.linalg.norm(v,axis=1).max());clamp=float(abs(newq[fixed]-q0[fixed]).max());row=dict(outer=k+1,original_native_R_N=R,drift_m=drift,algorithm_velocity_m_s=vmax,minimum_J=float(J.min()),clamp_drift_m=clamp);history.append(row)
    if not np.isfinite(newq).all() or J.min()<=.1 or clamp!=0:
     np.savez_compressed(OUT/(label+'-failed.npz'),positions_m=newq,velocities_m_s=v,J=J);raise ValueError('Nonfinite/inverted/movingclamp')
    q=newq;a,b=b,a
    if gates(R,drift,vmax,float(J.min()),clamp):break
   cpudiff=float(np.linalg.norm(q-ref['positions_m'],axis=1).max());pred=query(X,T,q,ref['material_probes_m']);querydiff=float(np.linalg.norm(pred-ref['probe_positions_m'],axis=1).max());passed=gates(R,drift,vmax,float(J.min()),clamp) and cpudiff<=20e-6 and querydiff<=20e-6
   precise=assemble(actual['local_force_N'],T,len(X));original64=float(np.linalg.norm((precise+force)[~fixed]));file=OUT/(label+'.npz');np.savez_compressed(file,rest_m=X,tets=T,faces=F,positions_m=q,velocities_m_s=v,fixed_node_ids=np.flatnonzero(fixed),readback_masses_kg=m,readback_inverse_mass=model.particle_inv_mass.numpy(),readback_poses=poses,readback_materials=mats,readback_gravity_m_s2=gravity,actual_native_elastic_forces_N=native,independent_readback_elastic_forces_N=precise,material_probes_m=ref['material_probes_m'],probe_positions_m=pred)
   row=dict(**j,completed=True,passed=bool(passed),outer_steps=len(history),**history[-1],native_rest_force_floor_N=restfloor,independent_readback_original_R_N=original64,maximum_CPU_allnode_error_m=cpudiff,maximum_CPU_query_error_m=querydiff,elapsed_s=time.monotonic()-start,history=history,capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)))
   s['cases'].append(row);fields[(j['E'],j['tets'],j['iterations'])]=pred;write(OUT/(label+'.json'),row);write(OUT/'summary.json',s)
   if not passed:raise RuntimeError('Staticqualification failed: '+label)
  delta=float(np.linalg.norm(fields[(300000,96,128)]-fields[(300000,96,512)],axis=1).max());s.update(completed=True,passed=all(r['passed'] for r in s['cases']) and delta<=20e-6,iteration_control=dict(maximum_query_difference_m=delta,passed=delta<=20e-6))
 except Exception as e:s.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:s['elapsed_s']=time.monotonic()-begin;write(OUT/'summary.json',s)
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol');p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r));raise SystemExit(0 if a.action=='prepare' or r['completed'] else 1)
