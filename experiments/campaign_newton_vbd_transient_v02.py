"""NEW finite-precision BDF2 momentum recovery with actual native positions.

Recovered velocity changes future helper history; native q is never projected.
D2q=v is checked through an explicit kinematic defect, not assumed exact.
Momentum residual is by construction and is not independent validation.
"""
from __future__ import annotations
import argparse, importlib.metadata, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
from experiments.campaign_newton_vbd_probe_v02 import ROOT, ENV, SITE, material, block_inputs, actual_input_reference, sha, write, EXTERNAL
from experiments.campaign_newton_vbd_static_v02 import query, assemble
from experiments.campaign_newton_static_initialization import accepted_start_solver_class
from experiments.campaign_newton_vbd_momentum import momentum_velocity,kinematic_defect,kinematic_position
STAGE='newton-vbd-transient-v02'
OP=ROOT/'runs/operator-dev'/STAGE
OUT=ROOT/'runs/gaussian-campaign-v01'/STAGE
CPU=ROOT/'runs/gaussian-campaign-v01/stable-transient-v01'
CPU_RELATIVE=ROOT/'runs/gaussian-campaign-v01/stable-transient-relative-v03'
def cpu_folder(hz):return CPU if hz==1920 else CPU_RELATIVE
JOBS=[('BDF2',hz,128) for hz in [1920,3840,7680]]+[('BDF2',3840,512)]
SOURCES=['experiments/campaign_newton_vbd_transient_v02.py','experiments/run_newton_vbd_transient_v02.py','tests/test_campaign_newton_vbd_transient_v02.py','experiments/campaign_newton_static_initialization.py','experiments/campaign_newton_vbd_static_kernels.py','experiments/campaign_newton_vbd_static_v02.py','experiments/campaign_newton_vbd_probe_v02.py','experiments/campaign_newton_vbd_kernels.py','experiments/boundary_block.py','experiments/campaign_newton_vbd_momentum.py','tests/test_campaign_newton_vbd_momentum.py']


def helper(method,step,h,q,v,oldq=None,oldv=None):
    if method not in ['BE','BDF2'] or step<1 or h<=0:
        raise ValueError('Declared method, positive step and timestep required')
    if method=='BE' or step==1:
        return h,np.asarray(v).copy()
    return 2*h/3,(np.asarray(q)-oldq)/(2*h)+(4*np.asarray(v)-oldv)/3


def physical_velocity(method,step,h,qnew,q,oldq=None):
    if method=='BE' or step==1:
        return (np.asarray(qnew)-q)/h
    return (3*(np.asarray(qnew)-q)-(np.asarray(q)-oldq))/(2*h)


def prepare():
    if (OP/'protocol.json').exists():raise FileExistsError('Preserve frozen protocol')
    s=json.loads((CPU/'summary.json').read_text())
    if not s['completed'] or not json.loads((CPU_RELATIVE/'summary.json').read_text())['completed']:raise ValueError('AllCPU references must finish first')
    refs={}
    for method,hz,_ in JOBS:
        key=f'{method}-hz{hz}'
        for extension in ['.npz','.json']:
            f=cpu_folder(hz)/(key+extension);refs[str(f.relative_to(ROOT))]=sha(f)
        row=json.loads((cpu_folder(hz)/(key+'.json')).read_text())
        if row.get('completed') is not True:raise ValueError('IncompleteCPUcase '+key)
    controls=ROOT/'runs/gaussian-campaign-v01/newton-vbd-static-v01/controls/summary.json'
    if not json.loads(controls.read_text())['passed']:raise ValueError('API controls required')
    p=dict(schema='Newton-physical-time-BDF2-momentum-protocol-v0.2',status='FROZEN_BEFORE_GPU_TRAJECTORIES',
        source_sha256={f:sha(ROOT/f) for f in SOURCES},external_source_sha256={f:sha(SITE/f) for f in EXTERNAL},
        CPU_reference_sha256=refs,CPU_protocol_sha256={f:sha(ROOT/f) for f in ['runs/operator-dev/stable-transient-v01/protocol.json','runs/operator-dev/stable-transient-relative-v03/protocol.json']},
        previous_GPU_result=dict(path='runs/gaussian-campaign-v01/newton-vbd-transient-v01/summary.json',sha256=sha(ROOT/'runs/gaussian-campaign-v01/newton-vbd-transient-v01/summary.json')),
        controls=dict(path=str(controls.relative_to(ROOT)),sha256=sha(controls)),CPU_threads=2,GPU=0,wall_cap_s=600,
        jobs=[dict(method=m,hz=h,iterations=i) for m,h,i in JOBS],duration_s=.1,
        physics=dict(size_m=[.24,.08,.08],cells=[4,2,2],tetrahedra=96,young_Pa=5000.,poisson=.35,density=625.,traction_N=1.,gravity=0.,physical_damping=0.,fixed='exact fullleftface',contacts=False),
        BDF2=dict(startup='one BE step',effective_dt='2h/3',helper_velocity='(qn-qprev)/(2h)+(4vn-vprev)/3',physical_velocity='Momentum:(4vn-vprev)/3+heff*(literalGPUinternal+actualF)/nativeMass;startupv=vold+h*force/nativeMass;fixedv0',chord_velocity_saved_separately='(3*(qnew-qn)-(qn-qprev))/(2h)',native_q_not_projected=True,momentum_residual_is_by_construction_NOT_independent_proof=True,objective='Sameelastic+M/(2heff²)||q-y||²-F.q; y=qn+heff*helperv'),
        gates=dict(maximum_allnode_CPU_position_error_m=20e-6,maximum_fixedquery_CPU_position_error_m=20e-6,maximum_128_to_512_iteration_position_difference_m=20e-6,minimum_J=.1,fixed_drift_m=0.,maximum_perstep_kinematic_defect_m=1e-7),
        nonclaims=['NativepositionsareNEVERoverwrittenbyvelocityintegration;qkinisdiagnosticonly','MomentumRbyconstructionisnotindependentproof;kinematicdefectandCPUcomparisonsareprimary','96tet discrete model only; no continuum qualification','No contact/rate-dependent damping support','GPU force residual and velocity discrepancy are diagnostics, not separately qualified','CPU temporal criterion and GPU approximation criterion are separate; no combined20um bound'])
    write(OP/'protocol.json',p)
    return dict(path=str((OP/'protocol.json').relative_to(ROOT)),sha256=sha(OP/'protocol.json'))


def validate(path,digest):
    if sha(ROOT/path)!=digest:raise ValueError('Changed protocol')
    p=json.loads((ROOT/path).read_text())
    for group,base in [('source_sha256',ROOT),('external_source_sha256',SITE),('CPU_reference_sha256',ROOT)]:
        if any(sha(base/f)!=h for f,h in p[group].items()):raise ValueError('Changed pinned input '+group)
    if any(sha(ROOT/f)!=v for f,v in p['CPU_protocol_sha256'].items()):raise ValueError('Changed CPU protocol')
    r=p['previous_GPU_result']
    if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Changed preservedv01')
    if sha(ROOT/p['controls']['path'])!=p['controls']['sha256']:raise ValueError('Changed controls')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('GPU0-only required')
    return p


def run(path,digest):
    p=validate(path,digest);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    state=dict(schema='Newton-physical-time-BDF2-momentum-results-v0.2',completed=False,passed=False,pid=os.getpid(),protocol=dict(path=path,sha256=digest),cases=[],nonclaims=p['nonclaims'])
    try:
        import warp as wp
        import newton
        from experiments.campaign_newton_vbd_static_kernels import elastic_local
        versions={x:importlib.metadata.version(x) for x in ['newton','warp-lang','numpy']}
        if versions['newton']!='1.7.0' or versions['warp-lang']!='1.18.0':raise ValueError('Pinned runtime required')
        wp.config.kernel_cache_dir=str(OUT/'warp-cache');wp.init();device=wp.get_device('cuda:0')
        # Persist own-device evidence before accepting any trajectory.
        probe=wp.zeros(1,device=device);wp.synchronize_device(device)
        uuids=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
        allowed=next(r.split(',')[1].strip() for r in uuids.splitlines() if r.split(',')[0].strip()=='0')
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
        own=[r for r in apps.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
        state.update(versions=versions,device_name=device.name,own_compute_records=own);write(OUT/'summary.json',state)
        if not own or any(r.split(',')[1].strip()!=allowed for r in own):raise ValueError('Wrong physical GPU assignment')
        mu,lam=material();solver_type=accepted_start_solver_class(newton.solvers.SolverVBD,wp);fields={}
        for method,hz,iterations in JOBS:
            label=f'{method}-hz{hz}-it{iterations}';begin=time.monotonic()
            with np.load(cpu_folder(hz)/f'{method}-hz{hz}.npz',allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
            X,T,F,mass,ids,force=block_inputs(True)
            expected_steps=round(.1*hz)
            if ref['positions_m'].shape!=(expected_steps+1,len(X),3) or ref['velocities_m_s'].shape!=ref['positions_m'].shape:
                raise ValueError('Unexpected CPU state shape')
            np.testing.assert_allclose(ref['times_s'],np.arange(expected_steps+1)/hz,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(X,ref['rest_m']);np.testing.assert_array_equal(T,ref['tets'])
            np.testing.assert_allclose(mass,ref['physical_masses_kg'],atol=1e-15,rtol=0)
            np.testing.assert_allclose(force,ref['applied_forces_N'],atol=2e-16,rtol=0)
            builder=newton.ModelBuilder(gravity=(0.,0.,0.))
            for j,(x,m) in enumerate(zip(X,mass)):
                builder.add_particle(tuple(x),(0.,0.,0.),float(m),radius=0.,flags=0 if j in ids else int(newton.ParticleFlags.ACTIVE))
            for t in T:builder.add_tetrahedron(*map(int,t),k_mu=mu,k_lambda=lam,k_damp=0.)
            builder.color();model=builder.finalize(device=device)
            np.testing.assert_array_equal(model.tet_indices.numpy(),T)
            np.testing.assert_array_equal(model.tet_materials.numpy(),np.broadcast_to(np.array([mu,lam,0],np.float32),(len(T),3)))
            np.testing.assert_allclose(model.particle_mass.numpy(),mass,rtol=1e-7)
            np.testing.assert_array_equal(model.particle_flags.numpy()[ids],0)
            if not np.all(model.particle_inv_mass.numpy()>0):raise ValueError('Physical mass missing')
            solver=solver_type(model,iterations=iterations,particle_enable_self_contact=False,particle_enable_tile_solve=False)
            a=model.state();b=model.state();control=model.control();fb=wp.array(force,dtype=wp.vec3,device=device)
            local=wp.zeros((len(T),4),dtype=wp.vec3,device=device)
            q=a.particle_q.numpy().astype(float);q0=q.copy();v=np.zeros_like(q);oldq=oldv=None
            positions=[q.copy()];velocities=[v.copy()];chord_velocities=[v.copy()];defects=[];ledger=[];targets=[];helper_velocities=[];h=1/hz;steps=round(.1*hz);qkin=q.copy();oldqkin=None;integrated_positions=[qkin.copy()]
            poses=model.tet_poses.numpy();mats=model.tet_materials.numpy();native_mass=model.particle_mass.numpy().astype(float);native_inv_mass=model.particle_inv_mass.numpy().astype(float);active=model.particle_flags.numpy()!=0
            for k in range(1,steps+1):
                heff,algorithm_v=helper(method,k,h,q,v,oldq,oldv)
                a.particle_qd.assign(wp.array(algorithm_v,dtype=wp.vec3,device=device));a.clear_forces();a.particle_f.assign(fb)
                solver.step(a,b,control,None,heff);wp.synchronize_device(device)
                qnew=b.particle_q.numpy().astype(float);chord_vnew=physical_velocity(method,k,h,qnew,q,oldq)
                wp.launch(elastic_local,dim=len(T)*4,inputs=[b.particle_q,model.tet_indices,model.tet_poses,model.tet_materials],outputs=[local],device=device);wp.synchronize_device(device)
                internal=assemble(local.numpy(),T,len(X));inertia=solver.inertia.numpy().astype(float);actual_force=fb.numpy().astype(float)
                vnew=momentum_velocity(method,k,h,v,oldv,internal,actual_force,native_mass,active)
                defect=kinematic_defect(method,k,h,qnew,q,oldq,vnew);newqkin=kinematic_position(method,k,h,qkin,oldqkin,vnew)
                kinematic_max=float(np.linalg.norm(defect,axis=1).max());velocity_defect=float(np.linalg.norm(chord_vnew-vnew,axis=1).max())
                accumulated_defect=float(np.linalg.norm(qnew-newqkin,axis=1).max())
                heff_native=float(np.float32(heff));implicit=native_mass[:,None]*(qnew-inertia)/heff_native**2-internal
                try:
                    precise=actual_input_reference(qnew,T,poses,float(mats[0,0]),float(mats[0,1]))
                except Exception:
                    np.savez_compressed(OUT/(label+'-failure.npz'),positions_m=qnew,velocities_m_s=vnew,step=k)
                    raise
                J=precise['J']
                acceleration=(vnew-v)/h if method=='BE' or k==1 else (3*vnew-4*v+oldv)/(2*h)
                internal64=assemble(precise['local_force_N'],T,len(X))
                physical_R=native_mass[:,None]*acceleration-internal64-fb.numpy().astype(float)
                constructed_R=native_mass[:,None]*acceleration-internal-actual_force
                quantum=native_mass[:,None]*abs(np.spacing(qnew.astype(np.float32)).astype(float))/heff_native**2
                b.particle_qd.assign(wp.array(vnew,dtype=wp.vec3,device=device))
                native_v=b.particle_qd.numpy()
                np.testing.assert_array_equal(native_v,vnew.astype(np.float32))
                np.testing.assert_array_equal(native_v[ids],0.)
                velocity_roundoff=float(np.linalg.norm(native_v.astype(float)-vnew,axis=1).max())
                targets.append(inertia.copy());helper_velocities.append(algorithm_v.copy())
                if not np.isfinite(qnew).all() or not np.isfinite(vnew).all() or J.min()<=.1 or not np.array_equal(qnew[ids],q0[ids]) or kinematic_max>1e-7:
                    np.savez_compressed(OUT/(label+'-failure.npz'),positions_m=qnew,velocities_m_s=vnew,chord_velocities_m_s=chord_vnew,kinematic_defect_m=defect,integrated_positions_m=newqkin,J=J,step=k)
                    raise RuntimeError('Nonfinite/inverted/moving-fixed/kinematic-defect transient;step='+str(k)+' maxdefect_m='+str(kinematic_max))
                ledger.append(dict(step=k,minimum_J=float(J.min()),literal_implicit_free_residual_N=float(np.linalg.norm(implicit[active])),physical_readback_free_residual_N=float(np.linalg.norm(physical_R[active])),nominal_one_ulp_inertial_force_scale_N=float(np.linalg.norm(quantum[active])),native_physical_velocity_roundoff_m_s=velocity_roundoff,momentum_constructed_free_residual_N=float(np.linalg.norm(constructed_R[active])),momentum_R_is_by_construction_NOT_independent_proof=True,maximum_chord_vs_momentum_velocity_defect_m_s=velocity_defect,maximum_position_kinematic_defect_m=kinematic_max,maximum_native_vs_integrated_velocity_position_difference_m=accumulated_defect))
                oldq,oldv=q,v;q,v=qnew,vnew;a,b=b,a;oldqkin,qkin=qkin,newqkin
                positions.append(q.copy());velocities.append(v.copy());chord_velocities.append(chord_vnew.copy());defects.append(defect.copy());integrated_positions.append(qkin.copy())
            positions=np.asarray(positions);velocities=np.asarray(velocities)
            node_error=float(np.linalg.norm(positions-ref['positions_m'],axis=2).max())
            sample_ids=np.arange(0,steps+1,hz//240)
            np.testing.assert_allclose(ref['query_times_s'],sample_ids/hz,atol=1e-12,rtol=0)
            pred=np.asarray([query(X,T,x,ref['material_probes_m']) for x in positions[sample_ids]])
            query_error=float(np.linalg.norm(pred-ref['query_positions_m'],axis=2).max())
            velocity_error=float(np.linalg.norm(velocities-ref['velocities_m_s'],axis=2).max())
            file=OUT/(label+'.npz')
            np.savez_compressed(file,times_s=np.arange(steps+1)/hz,positions_m=positions,velocities_m_s=velocities,
                query_times_s=ref['query_times_s'],query_positions_m=pred,material_probes_m=ref['material_probes_m'],rest_m=X,tets=T,faces=F,fixed_node_ids=ids,
                physical_masses_kg=mass,readback_poses=poses,readback_materials=mats,readback_mass=native_mass,readback_inverse_mass=native_inv_mass,chord_velocities_m_s=np.asarray(chord_velocities),kinematic_defects_m=np.asarray(defects),velocity_integrated_positions_m=np.asarray(integrated_positions),applied_forces_N=force,actual_applied_forces_N=fb.numpy(),native_inertial_targets_m=np.asarray(targets),algorithm_helper_velocities_m_s=np.asarray(helper_velocities))
            row=dict(method=method,hz=hz,iterations=iterations,nominal_physical_dt_s=h,native_effective_dt_float32_s=[float(np.float32(h)),float(np.float32(2*h/3))] if method=='BDF2' else[float(np.float32(h))],completed=True,passed=node_error<=20e-6 and query_error<=20e-6 and max(r['maximum_position_kinematic_defect_m'] for r in ledger)<=1e-7,maximum_position_kinematic_defect_m=max(r['maximum_position_kinematic_defect_m'] for r in ledger),maximum_chord_vs_momentum_velocity_defect_m_s=max(r['maximum_chord_vs_momentum_velocity_defect_m_s'] for r in ledger),maximum_native_vs_integrated_velocity_position_difference_m=max(r['maximum_native_vs_integrated_velocity_position_difference_m'] for r in ledger),maximum_mass_times_inverse_mass_minus1=float(np.max(abs(native_mass*native_inv_mass-1))),momentum_residual_by_construction_NOT_independent_proof=True,
                maximum_CPU_allnode_position_error_m=node_error,maximum_CPU_fixedquery_position_error_m=query_error,maximum_CPU_velocity_error_m_s=velocity_error,
                maximum_literal_implicit_free_residual_N=max(r['literal_implicit_free_residual_N'] for r in ledger),maximum_physical_readback_free_residual_N=max(r['physical_readback_free_residual_N'] for r in ledger),maximum_nominal_one_ulp_inertial_force_scale_N=max(r['nominal_one_ulp_inertial_force_scale_N'] for r in ledger),minimum_J=min(r['minimum_J'] for r in ledger),fixed_drift_m=0.,
                capture=dict(path=str(file.relative_to(ROOT)),sha256=sha(file)),elapsed_s=time.monotonic()-begin,ledger=ledger)
            state['cases'].append(row);write(OUT/(label+'.json'),row);write(OUT/'summary.json',state);fields[(method,hz,iterations)]=pred
        delta=float(np.linalg.norm(fields[('BDF2',3840,128)]-fields[('BDF2',3840,512)],axis=2).max())
        state['iteration_control']=dict(method='BDF2',hz=3840,iterations=[128,512],maximum_fixedquery_difference_m=delta,passed=delta<=20e-6)
        state.update(completed=True,passed=all(r['passed'] for r in state['cases']) and delta<=20e-6)
    except Exception as e:
        state.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        state['elapsed_s']=time.monotonic()-start;write(OUT/'summary.json',state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--protocol');p.add_argument('--protocol-sha256');a=p.parse_args()
    r=prepare() if a.action=='prepare' else run(a.protocol,a.protocol_sha256);print(json.dumps(r))
    if a.action=='run':raise SystemExit(0 if r['completed'] else 1)
