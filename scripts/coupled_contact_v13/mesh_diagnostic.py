"""Development-only spatial discretization audit; never changes frozen forecasts."""
from pathlib import Path
import sys,os,json,hashlib,time,argparse,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scripts.coupled_contact_v13.scenarios import run,build
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write

def interpolation(rest,tets,points):
    """Evaluate the actual P1 FEM basis, never an unrelated triangulation."""
    x=rest[tets];D=np.transpose(x[:,1:]-x[:,0,None],(0,2,1));inv=np.linalg.inv(D)
    W=np.zeros((len(points),len(rest)))
    for i,p in enumerate(points):
        b=np.einsum('tij,tj->ti',inv,p-x[:,0]);b=np.column_stack((1-b.sum(1),b))
        candidates=np.flatnonzero(np.all(b>=-1e-9,axis=1)&np.all(b<=1+1e-9,axis=1))
        if not len(candidates):raise ValueError('Point outside mesh')
        k=candidates[0];W[i,tets[k]]=b[k]
    if not np.allclose(W@rest,points,atol=1e-10):raise ValueError('Interpolation inconsistency')
    return W

def rotation(a,b):
    a=a-a.mean(0);b=b-b.mean(0);u,_,vt=np.linalg.svd(a.T@b);R=u@vt
    if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vt
    return R

def shape(a,b):
    ac=a-a.mean(0);bc=b-b.mean(0)
    return float(np.sqrt(np.mean((ac@rotation(a,b)-bc)**2)))

def task(spec):
    obj,name,E,mu,n=spec;dt=2.5e-5 if n==6 else 5e-5
    path=BASE/'mesh-diagnostic'/obj/name/f'n{n}';path.mkdir(parents=True,exist_ok=True)
    if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
    start=time.time()
    try:
        r=run(name,E,mu,(n,n,n),dt)
        system=build(name,E,mu,(n,n,n));query=build(name,E,mu,(2,2,2)).rest
        W=interpolation(r['rest'],r['tets'],query)
        sampled=np.einsum('qn,tnc->tqc',W,r['frames'])
        com=np.einsum('n,tnc->tc',system.mass/system.mass.sum(),r['frames'])
        R=rotation(query,sampled[-1]);nr=[shape(query,x) for x in sampled]
        np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},sampled=sampled,com=com)
        row=dict(object=obj,scenario=name,young=E,friction=mu,cells=n,dt=dt,success=True,seconds=time.time()-start,
                 min_j=r['min_j'],max_penetration_m=r['max_penetration_m'],friction_saturated_fraction=r['friction_saturated_fraction'],
                 max_kabsch_nonrigid_rms_m=max(nr),final_kabsch_nonrigid_rms_m=nr[-1],final_com_m=com[-1].tolist(),final_rotation=R.tolist(),path=str(path.relative_to(ROOT)))
    except Exception as e:row=dict(object=obj,scenario=name,cells=n,dt=dt,success=False,error=repr(e),seconds=time.time()-start)
    write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    sources=[ROOT/'scripts/coupled_contact_v13'/n for n in ('physics.py','scenarios.py','mesh_diagnostic.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    specs=[(obj,s,E,mu,n) for obj,E,mu in [('dev0',6000.,.35),('dev1',3000.,.1)] for s in ['edge_drop','curved'] for n in [2,3,4,6]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,a.workers)) as ex:rows=list(ex.map(task,specs))
    comparisons=[]
    for obj in ['dev0','dev1']:
        for name in ['edge_drop','curved']:
            group={r['cells']:r for r in rows if r['object']==obj and r['scenario']==name}
            for na,nb in [(2,4),(3,4),(4,6),(2,6),(3,6)]:
                ra,rb=group[na],group[nb];out=dict(object=obj,scenario=name,meshes=[na,nb],complete=ra['success'] and rb['success'])
                if out['complete']:
                    with np.load(ROOT/ra['path']/'trajectory.npz') as d:fa=d['sampled'];ca=d['com'];ta=d['times']
                    with np.load(ROOT/rb['path']/'trajectory.npz') as d:fb=d['sampled'];cb=d['com'];tb=d['times']
                    assert np.allclose(ta,tb,atol=1e-10,rtol=0)
                    A=np.array(ra['final_rotation']);B=np.array(rb['final_rotation']);angle=float(np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1))))
                    out.update(alltime_coordinate_rmse_m=float(np.sqrt(np.mean((fa[1:]-fb[1:])**2))),alltime_shape_rmse_m=float(np.sqrt(np.mean([shape(x,y)**2 for x,y in zip(fa[1:],fb[1:])]))),final_com_distance_m=float(np.linalg.norm(ca[-1]-cb[-1])),final_rotation_difference_deg=angle)
                comparisons.append(out)
    report=dict(scope='Post-freeze development-only spatial sensitivity diagnosis; not a replacement reference or benchmark result',source_hashes=hashes,source_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources),samples='27 common physical rest-lattice points evaluated with actual tetrahedral P1 barycentric interpolation; alltime metrics exclude t=0',timesteps='2/3/4 cells use 0.05ms;6 cells use0.025ms. Earlier4cells temporal qualification gives scale, but6cell timestep convergence not yet independently checked.',runs=rows,comparisons=comparisons)
    write(REPORT/'mesh-development-diagnostic.json',report)
if __name__=='__main__':main()
