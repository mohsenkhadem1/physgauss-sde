"""Predeclared development-only time, mesh and contact-quadrature qualification."""
from pathlib import Path
import sys,os,json,hashlib,argparse,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scripts.coupled_contact_v13.surface_scenarios import run,build
from scripts.coupled_contact_v13.surface_contact import SurfaceFEMSystem
from scripts.coupled_contact_v13.mesh_diagnostic import interpolation,shape
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write
NAMES=['squeeze','slide','moving_sphere','curved','edge_drop']

def key(spec):
    name,E,mu,n,dt,spacing=spec
    return f'{name}-E{E:g}-mu{mu:g}-n{n}-dt{dt:g}-spacing{spacing:g}'

def task(payload):
    spec,version=payload;name,E,mu,n,dt,spacing=spec;path=BASE/'surface-qualification'/version/key(spec);path.mkdir(parents=True,exist_ok=True)
    if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
    start=time.time()
    try:
        class Contact(SurfaceFEMSystem):surface_spacing=spacing
        r=run(name,E,mu,(n,n,n),dt,system_class=Contact)
        query=build(name,E,mu,(2,2,2)).rest;W=interpolation(r['rest'],r['tets'],query);sampled=np.einsum('qn,tnc->tqc',W,r['frames'])
        nr=[shape(query,x) for x in sampled]
        np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},sampled=sampled)
        out={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))}
        out.update(success=True,max_kabsch_deformation_m=max(nr),finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()))
    except Exception as e:out=dict(success=False,error=repr(e))
    out.update(spec=list(spec),key=key(spec),path=str(path.relative_to(ROOT)),seconds=time.time()-start)
    write(path/'status.json',out);print(json.dumps(out),flush=True);return out

def compare(a,b,sample=False):
    if not a['success'] or not b['success']:return dict(complete=False)
    with np.load(ROOT/a['path']/'trajectory.npz') as d:x=d['sampled' if sample else 'frames'];tx=d['times']
    with np.load(ROOT/b['path']/'trajectory.npz') as d:y=d['sampled' if sample else 'frames'];ty=d['times']
    if x.shape!=y.shape or not np.allclose(tx,ty,atol=1e-10,rtol=0):return dict(complete=False,reason='shape/time mismatch')
    return dict(complete=True,world_rms_m=float(np.sqrt(np.mean((x[1:]-y[1:])**2))),shape_rms_m=float(np.sqrt(np.mean([shape(i,j)**2 for i,j in zip(x[1:],y[1:])]))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    sources=[ROOT/'scripts/coupled_contact_v13'/n for n in ['physics.py','surface_contact.py','surface_scenarios.py','surface_qualification.py','mesh_diagnostic.py']]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    specs=[]
    for name in ['edge_drop','curved']:
        for E,mu in [(6000.,.35),(3000.,.1)]:
            specs.extend([(name,E,mu,4,5e-5,.003),(name,E,mu,6,2.5e-5,.003)])
    specs.extend((name,6000.,.35,4,dt,.003) for name in NAMES for dt in [1e-4,5e-5,2.5e-5])
    specs.extend((name,6000.,.35,4,5e-5,.0015) for name in ['edge_drop','curved'])
    specs=list(dict.fromkeys(specs))
    plan=dict(scope='Post-audit extension development only; no reserved outcomes',version=version,source_sha256=hashes,jobs=[list(s) for s in specs],gates=dict(penetration_m=.001,time_world_rms_m=.00025,time_refinement_ratio=.8,spatial_world_rms_m=.001,spatial_shape_rms_m=.0005,quadrature_world_rms_m=.001),spatial='4/6 cells on common27physicalP1points',time='Full nodal trajectories, firstframeexcluded',quadrature='3mm vs1.5mm material surface midpoint spacing at4cells',geometry='Surface extension box center x=.018m for edge vs edgecenter.01m; contactk19200 at.0004m²patch')
    planpath=REPORT/f'surface-qualification-plan-{version}.json'
    if not planpath.exists():write(planpath,plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,a.workers)) as ex:rows=list(ex.map(task,[(s,version) for s in specs]))
    bykey={r['key']:r for r in rows};checks=[]
    for name in ['edge_drop','curved']:
        for E,mu in [(6000.,.35),(3000.,.1)]:
            c=compare(bykey[key((name,E,mu,4,5e-5,.003))],bykey[key((name,E,mu,6,2.5e-5,.003))],sample=True)
            c.update(type='spatial',scenario=name,young=E,friction=mu);c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005;checks.append(c)
    for name in NAMES:
        coarse,medium,fine=[bykey[key((name,6000.,.35,4,dt,.003))] for dt in [1e-4,5e-5,2.5e-5]]
        c=compare(medium,fine);b=compare(coarse,fine);ratio=c.get('world_rms_m',np.inf)/max(b.get('world_rms_m',0),1e-20)
        c.update(type='temporal',scenario=name,refinement_ratio=ratio,coarse_to_finest_world_rms_m=b.get('world_rms_m'));c['passed']=c['complete'] and c['world_rms_m']<=.00025 and ratio<=.8;checks.append(c)
    for name in ['edge_drop','curved']:
        c=compare(bykey[key((name,6000.,.35,4,5e-5,.003))],bykey[key((name,6000.,.35,4,5e-5,.0015))])
        c.update(type='quadrature',scenario=name);c['passed']=c['complete'] and c['world_rms_m']<=.001;checks.append(c)
    for r in rows:r['physics_gate_passed']=r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders']
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
    report=dict(plan=plan,runs=rows,checks=checks,source_unchanged=unchanged,all_gates_pass=unchanged and all(c['passed'] for c in checks) and all(r['physics_gate_passed'] for r in rows))
    write(REPORT/f'surface-qualification-{version}.json',report);write(REPORT/'surface-development-qualification.json',report)
    print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
