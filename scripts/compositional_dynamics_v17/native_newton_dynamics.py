"""Unmodified Newton VBD dynamic controls; explicit P1/contact discretization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib,time,importlib.metadata
from pathlib import Path
import numpy as np
import warp as wp
import newton
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/compositional-dynamics-v17/native-newton-dynamics';REPORT=ROOT/'reports/compositional-dynamics-v17';PLAN=REPORT/'native-newton-dynamics-plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())

def surface_queries():
    uv=np.stack(np.meshgrid(np.linspace(0,1,17),np.linspace(0,1,17),indexing='ij'),-1);out=[]
    for face in range(6):
        q=np.zeros((17,17,3));axis=face//2;q[...,axis]=face%2;q[...,[j for j in range(3)if j!=axis]]=uv;out.append(q*[.12,.06,.06]+[0,-.03,-.03])
    return np.asarray(out)

def p1_map(rest,tets,query):
    # Float32 native restcoordinates need1e-6 barycentric tolerance at exact public boundary.
    # Keep signed weights for affine-exact interpolation; no large extrapolation permitted.
    D=np.stack([rest[tets[:,i]]-rest[tets[:,0]]for i in range(1,4)],-1);inv=np.linalg.inv(D);indices=[];weights=[]
    for point in np.asarray(query).reshape(-1,3):
        b=np.einsum('nij,nj->ni',inv,point-rest[tets[:,0]]);b=np.c_[1-b.sum(1),b];ok=np.flatnonzero(np.min(b,axis=1)>=-1e-6)
        if not len(ok):raise ValueError('Surface outside declared tetrahedra')
        j=ok[0];indices.append(tets[j]);weights.append(b[j])
    return np.asarray(indices),np.asarray(weights)

def save_prefix(out,job,times,states,vel,X,tets,mass,queries,frames,ledger,forces):
    tmp=out/'trajectory.pending.npz';path=out/'trajectory.npz'
    np.savez_compressed(tmp,times_s=times,positions=states,velocities=vel,rest=X,tets=tets,masses=mass,surface_rest=queries,surface_frames=frames,ledger=ledger,contact_wrench_on_soft_about_world_origin=forces)
    tmp.replace(path);write(out/'progress.json',dict(job=job,last_saved_time_s=float(times[-1]),steps_completed=len(ledger)-1,trajectory_sha256=sha(path),plan_sha256=sha(PLAN)))
    return path

def build(cells,case,device):
    E=12000.;nu=.35;mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));cells=np.array(cells,int)
    gravity=(0.,0.,-9.8)if case=='drop'else(0.,0.,0.)
    generator=newton.ModelBuilder(gravity=gravity)
    generator.add_soft_grid(pos=wp.vec3(0,-.03,-.03),rot=wp.quat_identity(),vel=wp.vec3(),dim_x=int(cells[0]),dim_y=int(cells[1]),dim_z=int(cells[2]),cell_x=.12/cells[0],cell_y=.06/cells[1],cell_z=.06/cells[2],density=625.,k_mu=mu,k_lambda=lam,k_damp=0.,particle_radius=0.)
    X=np.array(generator.particle_q,float);tets=np.array(generator.tet_indices,int)
    # Public native mesh API computes rhoV/4 masses; grid helper's rho*cellvolume per-node masses are not used.
    b=newton.ModelBuilder(gravity=gravity);b.add_soft_mesh(pos=wp.vec3(),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(),vertices=X.tolist(),indices=tets.ravel().tolist(),density=625.,k_mu=mu,k_lambda=lam,k_damp=0.,particle_radius=0.)
    ke=4e7*(.12*.06)/((cells[0]+1)*(cells[1]+1))
    if case=='drop':b.add_ground_plane(height=-.05,cfg=newton.ModelBuilder.ShapeConfig(ke=float(ke),kd=0.,mu=0.,margin=0.))
    b.color();m=b.finalize(device=device);m.soft_contact_ke=float(ke);m.soft_contact_kd=0.;m.soft_contact_mu=0.
    mass=m.particle_mass.numpy().astype(float)
    if abs(mass.sum()-.12*.06*.06*625)>1e-7:raise ValueError('Native known mass mismatch')
    return m,X,tets,mass,dict(mu_Pa=mu,lambda_Lame_Pa=lam,lambda_stable_NH_Pa=lam+mu,vertex_contact_ke_N_m=float(ke),mass_kg=float(mass.sum()),native_grid_helper_unused_mass_kg=float(sum(generator.particle_mass)))

def energy(x,v,rest,tets,mass,params,case):
    D=np.stack([rest[tets[:,i]]-rest[tets[:,0]]for i in range(1,4)],-1);F=np.stack([x[tets[:,i]]-x[tets[:,0]]for i in range(1,4)],-1)@np.linalg.inv(D);J=np.linalg.det(F)
    V=np.abs(np.linalg.det(D))/6;mu=params['mu_Pa'];lam=params['lambda_stable_NH_Pa'];alpha=1+mu/lam
    W=.5*mu*(F*F).sum((1,2))+.5*lam*(J-alpha)**2-(1.5*mu+.5*lam*(1-alpha)**2)
    depth=np.maximum(-.05-x[:,2],0)if case=='drop'else np.zeros(len(x))
    return np.array([.5*np.sum(mass[:,None]*v*v),float(V@W),float(np.sum(mass*9.8*(x[:,2]+.05)))if case=='drop'else 0.,.5*params['vertex_contact_ke_N_m']*float(depth@depth)]),float(J.min()),float(depth.max())

def source_inventory():
    site=Path(newton.__file__).parent
    paths=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/native_newton_run.py']+sorted((site/'_src').rglob('*.py'))
    return {str(p):sha(p)for p in paths}

def prepare():
    if PLAN.exists():raise ValueError('Preserve plan')
    jobs=[]
    for case in ['release','drop']:
        dts=[.0002,.0001,.00005]if case=='release'else[.0005,.00025,.000125]
        for dt in dts:jobs.append(dict(name=f'{case}-time-{dt}',case=case,cells=[4,2,2],dt=dt,iterations=32))
        for it in [16,64]:jobs.append(dict(name=f'{case}-iterations-{it}',case=case,cells=[4,2,2],dt=dts[-1],iterations=it))
        jobs.append(dict(name=f'{case}-mesh-8-4-4',case=case,cells=[8,4,4],dt=dts[-1],iterations=64))
    p=dict(scope='Native Newton VBD development forward controls, no heldout benchmark/superiority claim',jobs=jobs,duration_s={'release':.04,'drop':.25},sample_dt_s=.001,gravity_m_s2=9.8,density_kg_m3=625,young_Pa=12000,poisson=.35,physical_damping=0,friction=0,restitution='Not authored; rebound emerges from conservative penalty+bulk and native numerical dissipation',contact='Native vertex penalty, zero radius, meanbottomnode reference-area scaled4e7N/m3; not identicalQ2surfaceintegration. Same scalar onparticle+shape, fixedbeta0.',mass='Native add_soft_mesh rhoV/4 lumpedmass on native5tetgrid topology; different fromconsistentQ2mass',release='Initial20% zcompression, zerovelocity, nofloor, gravityzero,0.04s; matchesQ2developmentrelease',drop='Restgeometry, zero initialvelocity,20mm floor gap',solver=dict(rigid_compliant_alm=True,particle_enable_self_contact=False,particle_enable_tile_solve=False,rigid_avbd_beta=0.,rigid_soft_contact_use_log_barrier=False),collision=dict(soft_contact_gap=.002,enable_rigid_soft_full_surface_contact=False),qualification='Reportallfinite/J/penetration/energy/impulse and dt/iteration/mesh differences; prospectivecomparison thresholds world.25mm/shape.25mm, rebound2%, impacttime1ms. No automaticqualification ifmissingcomparison.',runtime=str(Path(__import__('sys').executable)),versions={n:importlib.metadata.version(n)for n in ['newton','warp-lang','numpy']},source_sha256=source_inventory(),max_job_seconds=900,max_stage_seconds=3600,max_parallel=2)
    write(PLAN,p)

def run(name,device='cpu',smoke_steps=None):
    p=read(PLAN)
    for path,h in p['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Sourcechanged '+path)
    if smoke_steps is None:
        review=read(REPORT/'native-newton-dynamics-prefit-independent-review.json')
        if review.get('passed')is not True or review.get('plan_sha256')!=sha(PLAN):raise ValueError('Exact developmentprotocol review required')
    elif smoke_steps>10:raise ValueError('Pre-review API preflight capped10steps')
    job=next(j for j in p['jobs']if j['name']==name);out=BASE/(name if smoke_steps is None else 'api-preflight')
    if out.exists():raise ValueError('Preserve results')
    out.mkdir(parents=True);t0=time.monotonic();wp.init();status=dict(job=job,plan_sha256=sha(PLAN),complete=False,device=device)
    try:
        m,X,tets,mass,params=build(job['cells'],job['case'],device);a=m.state();b=m.state();x=X.copy()
        if job['case']=='release':x[:,2]*=.8
        a.particle_q.assign(x.astype(np.float32));b.particle_q.assign(x.astype(np.float32))
        solver=newton.solvers.SolverVBD(m,iterations=job['iterations'],**p['solver']);pipeline=newton.CollisionPipeline(m,**p['collision']);contacts=pipeline.contacts();obs=solver.observables(['contact_f'],contacts=contacts);control=m.control()
        queries=surface_queries();ids,weights=p1_map(X,tets,queries);steps=round(p['duration_s'][job['case']]/job['dt'])if smoke_steps is None else smoke_steps;stride=round(p['sample_dt_s']/job['dt']);frames=[];states=[];vel=[];times=[];ledger=[];forces=[]
        for i in range(steps+1):
            q=a.particle_q.numpy().astype(float);v=a.particle_qd.numpy().astype(float);e,J,penetration=energy(q,v,X,tets,mass,params,job['case'])
            if not np.isfinite(q).all()or not np.isfinite(v).all()or J<=0:raise ValueError('Nonfinite or invertednative state')
            ledger.append([i*job['dt'],*e,J,penetration]);force=np.zeros(6)if i==0 else -obs.contact_f.numpy().sum(0);forces.append(force)
            if i%stride==0 or i==steps:times.append(i*job['dt']);states.append(q);vel.append(v);frames.append(np.einsum('nk,nkj->nj',weights,q[ids]).reshape(6,17,17,3))
            if i%stride==0 and (len(times)%25==1 or i==steps):save_prefix(out,job,times,states,vel,X,tets,mass,queries,frames,ledger,forces)
            if i==steps:break
            if time.monotonic()-t0>p['max_job_seconds']:raise TimeoutError('Bounded native job')
            a.clear_forces();pipeline.collide(a,contacts);solver.step(a,b,control,contacts,job['dt'],observables=obs);a,b=b,a
        path=save_prefix(out,job,times,states,vel,X,tets,mass,queries,frames,ledger,forces)
        ledger=np.asarray(ledger);forces=np.asarray(forces);momentum_change=np.sum(mass[:,None]*(vel[-1]-vel[0]),axis=0);contact_impulse=job['dt']*forces[1:,:3].sum(0)
        status.update(complete=True,parameters=params,trajectory_sha256=sha(path),nodes=len(X),tets=len(tets),minimum_J=float(ledger[:,5].min()),maximum_penetration_m=float(ledger[:,6].max()),initial_energy_J=float(ledger[0,1:5].sum()),final_energy_J=float(ledger[-1,1:5].sum()),contact_impulse_Ns=contact_impulse.tolist(),momentum_balance_residual_Ns=(momentum_change-contact_impulse-np.array([0,0,-9.8 if job['case']=='drop'else 0])*mass.sum()*steps*job['dt']).tolist(),force_convention='Nativecontact_f is shape reaction; negated first3forceN,last3torqueNm aboutworldorigin. Perstep finalconfiguration forces integratedrightendpoint.',colliders=[dict(type='plane',point=[0,0,-.05])]if job['case']=='drop'else[],case=job['case'],n=job['cells'],smoke_steps=smoke_steps)
    except Exception as e:
        status['error']=repr(e);status['failure_step']=locals().get('i');status['failure_time_s']=locals().get('i',0)*job['dt']
        if locals().get('states'):
            path=save_prefix(out,job,times,states,vel,X,tets,mass,queries,frames,ledger,forces);status.update(trajectory_sha256=sha(path),last_saved_time_s=float(times[-1]),partial_trajectory=True)
        else:status['initialization_or_initial_state_failure']=True
    status['wall_seconds']=time.monotonic()-t0;write(out/'complete.json',status);print(json.dumps(status))
    if not status['complete']:raise RuntimeError(status['error'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--name');parser.add_argument('--device',default='cpu');parser.add_argument('--smoke-steps',type=int);a=parser.parse_args();prepare()if a.stage=='prepare'else run(a.name,a.device,a.smoke_steps)
