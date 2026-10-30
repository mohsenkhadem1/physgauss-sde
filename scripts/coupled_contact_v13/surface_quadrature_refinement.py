"""Development-only contact quadrature ladder with conservative sparse evaluation."""
from pathlib import Path
import sys,os,json,hashlib,argparse,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scripts.coupled_contact_v13.sparse_surface_contact import SparseSurfaceFEMSystem
from scripts.coupled_contact_v13.surface_scenarios import run
from scripts.coupled_contact_v13.surface_qualification import compare
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write

def task(payload):
    name,spacing,version=payload;path=BASE/'surface-quadrature-refinement'/version/f'{name}-h{spacing:g}'
    if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
    path.mkdir(parents=True,exist_ok=True);start=time.time()
    try:
        class Contact(SparseSurfaceFEMSystem):surface_spacing=spacing
        r=run(name,6000.,.35,(4,4,4),5e-5,system_class=Contact)
        np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
        out={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))};out.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()))
    except Exception as e:out=dict(success=False,error=repr(e))
    out.update(scenario=name,spacing=spacing,path=str(path.relative_to(ROOT)),seconds=time.time()-start);write(path/'status.json',out);print(json.dumps(out),flush=True);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    names=['physics.py','surface_contact.py','sparse_surface_contact.py','surface_scenarios.py','surface_quadrature_refinement.py']
    sources=[ROOT/'scripts/coupled_contact_v13'/n for n in names];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    plan=dict(scope='Development-only quadrature refinement, same physical contact law; no fresh outcomes',source_sha256=hashes,version=version,cases=['edge_drop','curved'],young=6000.,friction=.35,cells=4,dt=.00005,spacing_ladder_m=[[.0015,.00075],[.00075,.000375]],gates=dict(world_rms_m=.001,shape_rms_m=.0005,penetration_m=.001),acceptance='Lower-cost spacing accepted only if BOTH cases pass both trajectory bounds and physical gates; second rung only if first fails',acceleration='Conservative SDF/Frobenius bound; exhaustive equivalence unit tests; same physical quadrature points')
    write(REPORT/f'surface-quadrature-plan-{version}.json',plan);rows={};checks=[];accepted=None
    for coarse,fine in plan['spacing_ladder_m']:
        missing=[(n,h,version) for n in plan['cases'] for h in (coarse,fine) if (n,h) not in rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,a.workers)) as ex:
            for r in ex.map(task,missing):rows[(r['scenario'],r['spacing'])]=r
        stage=[]
        for name in plan['cases']:
            aa,bb=rows[name,coarse],rows[name,fine];c=compare(aa,bb);c.update(scenario=name,spacings_m=[coarse,fine]);c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and all(r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders'] for r in (aa,bb));stage.append(c)
        checks.extend(stage);write(REPORT/f'surface-quadrature-progress-{version}.json',dict(plan=plan,runs=list(rows.values()),checks=checks));print(json.dumps(stage),flush=True)
        if all(c['passed'] for c in stage):accepted=coarse;break
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
    result=dict(plan=plan,runs=list(rows.values()),checks=checks,accepted_spacing_m=accepted,source_unchanged=unchanged,all_gates_pass=accepted is not None and unchanged)
    write(REPORT/f'surface-quadrature-refinement-{version}.json',result);write(REPORT/'surface-quadrature-refinement.json',result)
if __name__=='__main__':main()
