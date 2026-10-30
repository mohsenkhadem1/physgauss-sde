"""Development-only scenario diagnostics; never reads reserved objects."""
from pathlib import Path
import sys,json,time,argparse,traceback,concurrent.futures
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.coupled_contact_v13.scenarios import run
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write,remaining

def shape_error(a,b):
    a=a-a.mean(0);b=b-b.mean(0);u,_,vt=np.linalg.svd(a.T@b);r=u@vt
    if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
    return float(np.sqrt(np.mean((a@r-b)**2)))

def task(spec):
    name,E,mu,cells,dt=spec;key=f'{name}-E{E:g}-mu{mu:g}-n{cells[0]}-dt{dt:g}'
    out=BASE/'development'/key;out.mkdir(parents=True,exist_ok=True)
    t=time.monotonic()
    try:
        if remaining()<=0:raise RuntimeError('Campaign deadline')
        r=run(name,E,mu,cells,dt)
        np.savez_compressed(out/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
        d={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))}
        d.update(key=key,success=True,elapsed_seconds=time.monotonic()-t,max_shape_coordinate_rms_m=max(shape_error(r['frames'][0],x) for x in r['frames']),final_shape_coordinate_rms_m=shape_error(r['frames'][0],r['frames'][-1]))
    except Exception as e:
        d=dict(key=key,success=False,error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-t)
    write(out/'diagnostics.json',d);return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--refinement',action='store_true');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    cases=['squeeze','slide','edge_drop','curved','moving_sphere']
    specs=[(n,6000.,.35,(2,2,2),.0002) for n in cases]
    if a.refinement:
        specs=[(n,6000.,.35,(2,2,2),dt) for n in cases for dt in [.0004,.0002,.0001]]+[(n,6000.,.35,(4,4,4),dt) for n in cases for dt in [.0001,.00005]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(task,specs))
    write(REPORT/('development-refinement.json' if a.refinement else 'development-scenarios.json'),dict(scope='Development only; parameters E6000Pa friction0.35',cases=rows))
    for r in rows:print(json.dumps(r,default=str),flush=True)
if __name__=='__main__':main()
