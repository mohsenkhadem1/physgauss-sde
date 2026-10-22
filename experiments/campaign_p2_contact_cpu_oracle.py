"""Independent sparse CPU oracle, exactly one declared six-tet compliant drop.

Bulk/consistent mass/BDF2 reused unchanged. Surface trace, quadrature and contact
sparse gradient/tangent assembled here in NumPy/SciPy, no GPU contact calls.
"""
import argparse,json,multiprocessing,os,signal,time
from pathlib import Path
import numpy as np
from scipy import sparse
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_consistent_transient import ConsistentP2Transient,scalar_mass
from experiments.campaign_newton_stable_material import StableP2Block
from experiments.campaign_static_p2_block import shape_gradients,EDGES
from experiments.run_static_p2_axial_continuation import sha,dump,record,watchdog


def surface_geometry(m,order=8):
    edges={tuple(sorted((int(T[a]),int(T[b])))):int(e[4+k]) for T,e in zip(m.tets,m.elements) for k,(a,b) in enumerate(EDGES)}
    faces=np.array([list(f)+[edges[tuple(sorted(map(int,f[[a,b]])))] for a,b in [(0,1),(0,2),(1,2)]] for f in m.faces],dtype=int)
    X=m.rest[m.faces];area=np.linalg.norm(np.cross(X[:,1]-X[:,0],X[:,2]-X[:,0]),axis=1)/2
    x,w=leggauss(order);x=(x+1)/2;w=w/2;N=[];W=[]
    for i,a in enumerate(x):
        for j,b in enumerate(x):
            lam=np.array([(1-a)*(1-b),a,(1-a)*b]);N.append(np.r_[lam*(2*lam-1),4*lam[0]*lam[1],4*lam[0]*lam[2],4*lam[1]*lam[2]]);W.append(2*w[i]*w[j]*(1-a))
    return faces,area,np.array(N),np.array(W)


def surface_response(q,faces,area,N,w,kappa=1e9,floor=0.,tangent=False):
    gap=np.einsum('qa,fa->qf',N,q[faces,2])-floor;negative=np.minimum(gap,0);weighted=w[:,None]*area[None];energy=.5*kappa*np.sum(weighted*negative**2)
    gz=kappa*np.einsum('qf,qa->fa',weighted*negative,N);g=np.zeros_like(q);np.add.at(g[:,2],faces.ravel(),gz.ravel());H=None
    if tangent:
        gram=kappa*np.einsum('qf,qa,qb->fab',weighted*(gap<0),N,N);dofs=3*faces+2;H=sparse.coo_matrix((gram.ravel(),(np.repeat(dofs,6,1).ravel(),np.tile(dofs,(1,6)).ravel())),shape=(q.size,q.size)).tocsr()
    return float(energy),g,H,float(max(0.,-gap.min()))


class CPUContactOracle(ConsistentP2Transient):
    def __init__(self):
        m=StableP2Block((1,1,1),young=300000.,poisson=.35,density=625.,gravity=9.81,traction=0.,order=3)
        m.mask[:]=False;m.free=np.arange(m.rest.size);m.rest=m.rest+np.array([0.,0.,.09]);self.model=m;self.mass=scalar_mass(m)
        self.grad=np.stack([shape_gradients(b[None],m.linear_gradients) for b in m.bary]);self.check_grad=np.stack([shape_gradients(b[None],m.linear_gradients) for b in m.check_bary]);self.weights=m.weights[:,None]*m.volumes[None];self.surface=surface_geometry(m)
    def elastic_u(self,u,tangent=False):
        E,g,H,J=super().elastic_u(u,tangent);U,gs,Hs,_=surface_response(self.model.rest+u,*self.surface,tangent=tangent)
        return E+U,g+gs,None if H is None else H+Hs,J


def prepare(path):
    gp=Path('runs/gaussian-campaign-v01/p2-contact-validation-v01/drop6_hz480.json');row=json.loads(gp.read_text())
    if not row['completed'] or sha(row['capture']['path'])!=row['capture']['sha256']:raise ValueError('Completed hashed GPU target required')
    sources=['experiments/campaign_p2_contact_cpu_oracle.py','experiments/campaign_p2_consistent_transient.py','experiments/campaign_newton_stable_material.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_supported_block.py','experiments/campaign_static_uniaxial_bar.py','experiments/campaign_stable_transient.py','experiments/run_static_p2_axial_continuation.py','experiments/boundary_block.py']
    p=dict(schema='C38-independent-CPU-P2-contact-oracle-v0.1',status='FROZEN_BEFORE_CPU_DROP',source_sha256={f:sha(f) for f in sources},GPU_target_json=record(gp),GPU_target_capture=row['capture'],GPU_protocol=record(Path('runs/operator-dev/p2-contact-validation-v01/protocol.json')),CPU_threads=2,hard_wall_cap_s=120,actual_RSS_guard_bytes=8*1024**3,forecast_s=[10,30],physics=dict(cells=[1,1,1],E_Pa=300000.,nu=.35,rho=625,gravity_world_m_s2=[0,0,-9.81],bottom_gap_m=.05,initial='REST and v0',duration_s=.15,hz=480,all_nodes_free=True,kappa_N_m3=1e9,plane_z=0.,surface_Q=8,friction=0,damping=0,law='Finite reference-area wholeboundary P2 plane penalty'),scheme='Unchanged full-consistent-T10 BE-start BDF2 and exact directu stableNH sparse Newton',gates=dict(original_objective_R_N=1e-10,physical_R_N=1e-9,minimum_J=.1,allnode_CPU_GPU_max_position_m=1e-8,allnode_CPU_GPU_max_velocity_m_s=1e-5),shared_components=['Frozen stableNH constitutive function and P2 reference geometry','Frozen analytic full consistent mass and BDF2 advance','Same mathematical positive8x8 Gauss-Duffy triangle rule, independently generated/assembled here'],independent_components=['NumPy/SciPy contact gradient/full sparseH; no Torch contact plugin calls','Direct sparse original Newton linear solves vsGPU PCG','Separate surface connectivity and reference-area construction'],nonclaims=['Same discrete contact-law/time validation only; not time/space/penalty/nonpenetration qualification','All original native drop-refinement and kappa failures retained','No new E/kappa/force/time tuning or retry'])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    dump(path,p)


def run(path,digest,output):
    if sha(path)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(path).read_text())
    for f,h in p['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    for k in ['GPU_target_json','GPU_target_capture','GPU_protocol']:
        if sha(p[k]['path'])!=p[k]['sha256']:raise ValueError('Changed GPU evidence')
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('Two CPU threads required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();r=dict(completed=False,passed=False,protocol=record(path));states=[];ledger=[]
    def save():r['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',r)
    def interrupt(s,f):raise TimeoutError('Frozen120s/8GiB resource cap')
    signal.signal(signal.SIGTERM,interrupt);guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(os.getpid(),start+120,8*1024**3,out/'resource-guard.json'),daemon=True);guard.start()
    try:
        c=CPUContactOracle();state=c.initial();previous=None;states=[state]
        for n in range(72):
            new,d=c.step(state,previous,1/480,c.model.load);d.update(step=n+1,time_s=(n+1)/480);ledger.append(d)
            if d['objective_free_R_N']>1e-10 or d['physical_original_free_R_N']>1e-9 or d['minimum_J']<=.1:raise ValueError('Original residual/J gate failed')
            previous,state=state,new;states.append(state)
        U=np.stack([s.u for s in states]);V=np.stack([s.v for s in states]);Q=c.model.rest+U;times=np.arange(73)/480
        with np.load(p['GPU_target_capture']['path'],allow_pickle=False) as a:
            for k,v in [('rest_m',c.model.rest),('tets',c.model.tets),('elements',c.model.elements),('faces',c.model.faces)]:np.testing.assert_array_equal(a[k],v)
            clock=float(np.max(abs(times-a['times_s'])));qerr=float(np.linalg.norm(Q-a['positions_m'],axis=-1).max());verr=float(np.linalg.norm(V-a['velocities_m_s'],axis=-1).max())
        f=out/'CPU-drop6-hz480.npz';np.savez_compressed(f,times_s=times,rest_m=c.model.rest,tets=c.model.tets,elements=c.model.elements,faces=c.model.faces,positions_m=Q,displacements_m=U,velocities_m_s=V,step_increments_m=np.stack([s.delta for s in states]),gravity_force_N=c.model.load)
        r.update(completed=True,passed=qerr<=1e-8 and verr<=1e-5 and clock<=1e-15,steps=72,maximum_CPU_GPU_position_difference_m=qerr,maximum_CPU_GPU_velocity_difference_m_s=verr,maximum_clock_difference_s=clock,maximum_objective_R_N=max(d['objective_free_R_N'] for d in ledger),maximum_physical_R_N=max(d['physical_original_free_R_N'] for d in ledger),minimum_J=min(d['minimum_J'] for d in ledger),capture=record(f),space_time_penalty_or_training_qualified=False)
    except Exception as e:
        r.update(error=repr(e),failure_history=getattr(e,'history',[]))
        if states:np.savez_compressed(out/'last-accepted-state.npz',u=states[-1].u,v=states[-1].v,delta=states[-1].delta)
    finally:dump(out/'steps.json',ledger);save();guard.terminate();guard.join(timeout=2)
    return r
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        r=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(r,indent=2));raise SystemExit(0 if r['completed'] else 1)
