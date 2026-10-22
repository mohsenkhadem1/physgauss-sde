"""Standalone VBD controls and original-force static relaxation; not dynamics.

STATIC: stockbackwardEuler, pseudo h1s, resetalgorithmvelocity beforeeachstep.
No physical damping/stiffness/force change; stationarity checked without inertia.
"""
from __future__ import annotations
import argparse,json,os,time,traceback,subprocess
from pathlib import Path
import numpy as np
from experiments.campaign_newton_vbd_probe_v02 import ROOT,ENV,SITE,material,block_inputs,actual_input_reference,sha,write,EXTERNAL
PROTOCOL_ROOT='runs/operator-dev/newton-vbd-static-v01';OUTPUT_ROOT='runs/gaussian-campaign-v01/newton-vbd-static-v01'
SOURCES=['experiments/campaign_newton_vbd_static.py','experiments/campaign_newton_vbd_static_kernels.py','experiments/campaign_newton_vbd_probe_v02.py','experiments/campaign_newton_vbd_kernels.py','experiments/boundary_block.py','tests/test_campaign_newton_vbd_static.py','experiments/run_newton_vbd_static.py']


def assemble(local,tets,count):
 out=np.zeros((count,3));np.add.at(out,np.asarray(tets),np.asarray(local));return out


def query(rest,tets,positions,points):
 X=np.asarray(rest);T=np.asarray(tets);P=np.asarray(points);corner=X[T];inverse=np.linalg.inv(np.swapaxes(corner[:,1:]-corner[:,:1],-1,-2))
 xyz=np.einsum('eij,pej->pei',inverse,P[:,None]-corner[None,:,0]);bary=np.concatenate([1-xyz.sum(2)[:,:,None],xyz],axis=2);inside=np.all(bary>=-2e-12,2)&np.all(bary<=1+2e-12,2)
 if not inside.any(1).all():raise ValueError('Fixedmaterialquery outside authoredrestmesh')
 ids=inside.argmax(1);b=bary[np.arange(len(P)),ids];return np.einsum('pa,pai->pi',b,np.asarray(positions)[T[ids]])


def static_gates(residual,drift,query_delta,clamp,J,algorithm_velocity):
 return dict(algorithm_velocity_passed=bool(np.isfinite(algorithm_velocity) and algorithm_velocity<=.5e-6),original_force_passed=bool(residual<=1e-3),stationary_passed=bool(drift<=.5e-6),matched_query_passed=bool(query_delta<=20e-6),fixed_passed=bool(clamp==0),positive_J_passed=bool(J>.1),passed=bool(np.isfinite(algorithm_velocity) and algorithm_velocity<=.5e-6 and residual<=1e-3 and drift<=.5e-6 and query_delta<=20e-6 and clamp==0 and J>.1))


def prepare(mode):
 if mode not in ['controls','static']:raise ValueError('Explicitmode required')
 p=ROOT/PROTOCOL_ROOT/(mode+'-protocol.json')
 if p.exists():raise FileExistsError('Retainfrozenstage')
 ref=ROOT/'runs/gaussian-campaign-v01/newton-material-alignment-v01/P1-tets96.npz';meta=ref.with_suffix('.json')
 m=json.loads(meta.read_text())
 if not m['completed']:raise ValueError('CompletedmatchedCPUreference required')
 d=dict(schema='newton-VBD-original-force-static-v0.1',status='FROZEN_BEFORE_STAGE_CALLS',mode=mode,source_sha256={s:sha(ROOT/s) for s in SOURCES},external_source_sha256={s:sha(SITE/s) for s in EXTERNAL},CPU_threads=2,GPU=0,hard_cap_s=600,
  matched_CPU_reference=dict(path=ref.relative_to(ROOT).as_posix(),sha256=sha(ref)),matched_CPU_metadata=dict(path=meta.relative_to(ROOT).as_posix(),sha256=sha(meta)),
  previous_kernel_controls=[dict(path='runs/gaussian-campaign-v01/newton-vbd-probe-'+v+'/summary.json',sha256=sha(ROOT/('runs/gaussian-campaign-v01/newton-vbd-probe-'+v+'/summary.json'))) for v in ['v01','v02']],
  nearzero_kernel_parity_status='UNRESOLVED_AT2uN;notrelabelledPASS;doesnotblockindependentstaticgates',
  material=dict(E_Pa=5000.,nu=.35,rho_kg_m3=625.,damping=0.,law='Newton1.7stableNH,kappa=lambdaL+muLinternally'),mesh=dict(cells=[4,2,2],size_m=[.24,.08,.08],mass='ExactP1physicaltetlumping;retainedfixednodemass'),
  controls=dict(freefall_steps=12,physics_hz=240,iterations=16,gravity_m_s2=[0,0,-9.81],force_one_step_N=1.,velocity_max_error_m_s=2e-4,translation_nonuniformity_m=2e-5),
  static=dict(iterations=[32,128,512],all_from_REST=True,maximum_outer_steps=200,pseudo_dt_s=1.,reset_algorithm_velocity_each_outerstep=True,physical_damping=0.,stop_using_only_original_R_and_drift=True,
  gates=dict(original_literal_free_force_residual_N=1e-3,maximum_consecutive_q_drift_m=.5e-6,maximum_algorithm_velocity_m_per_pseudosecond=.5e-6,matched122fixedquery_maxerror_m=20e-6,fixed_clamp_error_m=0.,minimum_J=.1)),
  qualification='96tetdiscreteSTATICequilibriumcomparisononly;notspatial/physicaltransient/time/contactqualification;nearzerokernelfloorreportedexplicitly')
 write(p,d);return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))


def validate(path,digest):
 if sha(ROOT/path)!=digest:raise ValueError('Changed stageprotocol')
 d=json.loads((ROOT/path).read_text())
 if d['source_sha256']!={s:sha(ROOT/s) for s in SOURCES} or d['external_source_sha256']!={s:sha(SITE/s) for s in EXTERNAL}:raise ValueError('Changedsourceclosure')
 for key in ['matched_CPU_reference','matched_CPU_metadata']:
  r=d[key]
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('ChangedCPUreference')
 for r in d['previous_kernel_controls']:
  if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changedpreservedfailure')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or d['hard_cap_s']!=600 or d['GPU']!=0:raise ValueError('Device/deadlinechanged')
 return d


def run(path,digest):
 d=validate(path,digest);mode=d['mode'];out=ROOT/OUTPUT_ROOT/mode
 if out.exists():raise FileExistsError('Freshstageoutputrequired')
 out.mkdir(parents=True);start=time.monotonic();state=dict(completed=False,passed=False,mode=mode,pid=os.getpid(),protocol=dict(path=path,sha256=digest),variants=[],qualification=d['qualification'],nearzero_kernel_parity_status=d['nearzero_kernel_parity_status'])
 try:
  import warp as wp
  import newton,importlib.metadata
  from experiments.campaign_newton_vbd_static_kernels import elastic_local
  wp.config.kernel_cache_dir=str(ROOT/OUTPUT_ROOT/'warp-cache');wp.init();device=wp.get_device('cuda:0');mu,lam=material()
  state['versions']={p:importlib.metadata.version(p) for p in ['newton','warp-lang','numpy']};state['device_name']=device.name
  def build(iterations,fixed,gravity):
   X,T,F,mass,ids,force=block_inputs(fixed);builder=newton.ModelBuilder(gravity=tuple(gravity))
   for i,(x,m) in enumerate(zip(X,mass)):builder.add_particle(tuple(x),(0.,0.,0.),float(m),radius=0.,flags=0 if i in ids else int(newton.ParticleFlags.ACTIVE))
   for t in T:builder.add_tetrahedron(*map(int,t),k_mu=mu,k_lambda=lam,k_damp=0.)
   builder.color();model=builder.finalize(device=device);poses=model.tet_poses.numpy().copy();mats=model.tet_materials.numpy().copy();flags=model.particle_flags.numpy();readmass=model.particle_mass.numpy()
   np.testing.assert_array_equal(model.tet_indices.numpy(),T);np.testing.assert_array_equal(mats,np.broadcast_to(np.array([mu,lam,0],np.float32),(len(T),3)));np.testing.assert_allclose(readmass,mass,rtol=1e-7);np.testing.assert_array_equal(flags[ids],0);np.testing.assert_array_equal(flags[flags!=0],int(newton.ParticleFlags.ACTIVE));np.testing.assert_array_equal(model.particle_q.numpy(),X.astype(np.float32));np.testing.assert_array_equal(model.gravity.numpy().reshape(-1,3),np.array(gravity,dtype=np.float32)[None])
   if not np.all(readmass>0) or not np.all(model.particle_inv_mass.numpy()>0):raise RuntimeError('Physicalmasslost')
   solver=newton.solvers.SolverVBD(model,iterations=iterations,particle_enable_self_contact=False,particle_enable_tile_solve=False);a=model.state();b=model.state();control=model.control();local=wp.zeros((len(T),4),dtype=wp.vec3,device=device);fb=wp.array(force,dtype=wp.vec3,device=device)
   return model,solver,a,b,control,X,T,F,mass,ids,force,fb,local,poses,mats,flags
  def audit(model,a,local,T,X,poses,mats,force,flags):
   wp.launch(elastic_local,dim=len(T)*4,inputs=[a.particle_q,model.tet_indices,model.tet_poses,model.tet_materials],outputs=[local],device=device);wp.synchronize_device(device)
   q=a.particle_q.numpy().astype(float);forces=assemble(local.numpy(),T,len(X));r=forces+np.asarray(force);active=flags!=0;precise=actual_input_reference(q,T,poses,float(mats[0,0]),float(mats[0,1]));pr=assemble(precise['local_force_N'],T,len(X))+force
   return q,float(np.linalg.norm(r[active])),float(np.linalg.norm(pr[active])),float(precise['J'].min()),forces
  if mode=='controls':
   for fixed,steps,gravity in [(False,12,(0,0,-9.81)),(True,1,(0,0,0))]:
    model,solver,a,b,c,X,T,F,mass,ids,force,fb,local,poses,mats,flags=build(16,fixed,gravity);q0=a.particle_q.numpy().astype(float);h=1/240
    for k in range(steps):
     a.clear_forces()
     if fixed:a.particle_f.assign(fb)
     solver.step(a,b,c,None,h);wp.synchronize_device(device);a,b=b,a
    q=a.particle_q.numpy().astype(float);v=a.particle_qd.numpy().astype(float)
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise RuntimeError('Nonfinitecontrol')
    disp=q-q0;mean=mass@disp/mass.sum();spread=float(np.sqrt(np.mean((disp-mean)**2)));verror=float(np.max(abs(v-steps*h*np.asarray(gravity))))
    _,r,r64,J,_=audit(model,a,local,T,X,poses,mats,force if fixed else mass[:,None]*gravity,flags)
    exactfixed=not len(ids) or np.array_equal(q[ids],q0[ids]);passed=exactfixed and J>.1 and (fixed or(verror<=2e-4 and spread<=2e-5))
    row=dict(kind='force_one_step' if fixed else'freefall',steps=steps,passed=passed,velocity_max_error_m_s=verror if not fixed else None,translation_nonuniformity_coordinate_RMS_m=spread,minimum_J=J,fixed_exact=exactfixed,static_original_R_not_equilibriumclaim_N=r,readback_float64_static_R_N=r64,physical_mass_kg=float(mass.sum()),readback_mass_kg=float(model.particle_mass.numpy().sum()))
    np.savez_compressed(out/(row['kind']+'.npz'),rest_m=X,tets=T,faces=F,positions_m=q,velocities_m_s=v,readback_poses=poses,readback_materials=mats,readback_mass=model.particle_mass.numpy(),readback_flags=flags,fixed_node_ids=ids);state['variants'].append(row);write(out/'summary.json',state)
    if not passed:raise RuntimeError('Independentcontrolfailed')
  else:
   with np.load(ROOT/d['matched_CPU_reference']['path'],allow_pickle=False) as f:CPU={k:f[k].copy() for k in ['rest_m','tets','applied_forces_N','material_probes_m','probe_positions_m','positions_m']}
   for iterations in [32,128,512]:
    model,solver,a,b,c,X,T,F,mass,ids,force,fb,local,poses,mats,flags=build(iterations,True,(0,0,0));q0=a.particle_q.numpy().astype(float);np.testing.assert_array_equal(X,CPU['rest_m']);np.testing.assert_array_equal(T,CPU['tets']);np.testing.assert_allclose(force,CPU['applied_forces_N'],atol=2e-16,rtol=0)
    force_actual=fb.numpy().astype(float);q,R0,R064,J,_=audit(model,a,local,T,X,poses,mats,np.zeros_like(force),flags);ledger=[];row=dict(iterations=iterations,from_REST=True,rest_literal_free_elastic_force_N=R0,rest_readback_float64_free_elastic_force_N=R064)
    for outer in range(1,201):
     old=a.particle_q.numpy().astype(float);a.particle_qd.zero_();a.clear_forces();a.particle_f.assign(fb);solver.step(a,b,c,None,1.);wp.synchronize_device(device);a,b=b,a
     q,R,R64,J,internal=audit(model,a,local,T,X,poses,mats,force_actual,flags);drift=float(np.linalg.norm(q-old,axis=1).max());clamp=float(abs(q[ids]-q0[ids]).max());vmax=float(np.linalg.norm(a.particle_qd.numpy(),axis=1).max())
     if not np.isfinite(q).all() or J<=.1 or clamp!=0:raise RuntimeError('Nonfinite/inverted/movingfixedstaticstate')
     ledger.append(dict(outer_step=outer,original_literal_free_R_N=R,readback_float64_free_R_N=R64,q_drift_max_m=drift,algorithm_velocity_max_m_per_pseudosecond=vmax,minimum_J=J))
     if outer%10==0:write(out/'progress.json',dict(iterations=iterations,outer_step=outer,original_R_N=R,q_drift_max_m=drift,elapsed_s=time.monotonic()-start))
     if R<=1e-3 and drift<=.5e-6 and vmax<=.5e-6:break
    # CPUquery used ONLYafteralgorithmstops/caps; it neverselectsiterations/steps.
    pred=query(X,T,q,CPU['material_probes_m']);delta=float(np.linalg.norm(pred-CPU['probe_positions_m'],axis=1).max());gates=static_gates(R,drift,delta,clamp,J,vmax)
    row.update(actual_applied_force_sum_N=force_actual.sum(0).tolist(),completed=True,outer_steps=outer,capped=outer==200,original_literal_free_R_N=R,readback_float64_free_R_N=R64,q_drift_max_m=drift,algorithm_velocity_max_m_per_pseudosecond=vmax,maximum_CPU_query_error_m=delta,maximum_CPU_node_error_m=float(np.linalg.norm(q-CPU['positions_m'],axis=1).max()),minimum_J=J,gates=gates,ledger=ledger)
    np.savez_compressed(out/('iterations'+str(iterations)+'.npz'),rest_m=X,tets=T,faces=F,positions_m=q,reference_CPU_positions_m=CPU['positions_m'],material_probes_m=CPU['material_probes_m'],probe_positions_m=pred,readback_rest_positions_m=q0,readback_poses=poses,readback_materials=mats,readback_mass=model.particle_mass.numpy(),readback_flags=flags,actual_applied_force_N=force_actual,final_literal_internal_force_N=internal,fixed_node_ids=ids)
    state['variants'].append(row);write(out/'summary.json',state)
  gpuinfo=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;gpu0=gpuinfo.splitlines()[0].split(',')[1].strip();apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout;own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
  if not own or any(r.split(',')[1].strip()!=gpu0 for r in own):raise RuntimeError('ComputeassignmentnotsoleGPU0')
  state['own_compute_records']=own
  state.update(completed=True,passed=all(v.get('passed',v.get('gates',{}).get('passed',False)) for v in state['variants']))
 except Exception as e:state.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:state['elapsed_s']=time.monotonic()-start;write(out/'summary.json',state)
 return state

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--mode',choices=['controls','static']);p.add_argument('--protocol');p.add_argument('--protocol-sha256');a=p.parse_args();r=prepare(a.mode) if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r))
 if a.action=='run':raise SystemExit(0 if r['completed'] else 1)
