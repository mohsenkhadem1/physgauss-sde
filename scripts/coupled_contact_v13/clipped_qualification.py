"""Development qualification of exact clipping and remaining friction/sphere rules."""
from pathlib import Path
import sys,os,json,hashlib,argparse,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scripts.coupled_contact_v13.clipped_surface_contact import ClippedSurfaceFEMSystem,load_kernel
from scripts.coupled_contact_v13.surface_scenarios import run
from scripts.coupled_contact_v13.surface_qualification import compare
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write,remaining
CASES=[(6000.,.35),(3000.,.1)]

def key(s):
    n,E,mu,dt,h,level=s
    return f'{n}-E{E:g}-mu{mu:g}-dt{dt:g}-h{h:g}-l{level}'

def task(payload):
    spec,version=payload;name,E,mu,dt,h,level=spec;path=BASE/'clipped-qualification'/version/key(spec)
    if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
    path.mkdir(parents=True,exist_ok=True);start=time.time()
    try:
        if remaining()<=0:raise RuntimeError('Campaign deadline reached')
        class Contact(ClippedSurfaceFEMSystem):surface_spacing=h;friction_quadrature_level=level
        r=run(name,E,mu,(4,4,4),dt,system_class=Contact)
        np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
        row={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))};row.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()))
    except Exception as e:row=dict(success=False,error=repr(e))
    row.update(spec=list(spec),key=key(spec),path=str(path.relative_to(ROOT)),seconds=time.time()-start);write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def physics(r):return r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders']

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','surface_scenarios.py','clipped_qualification.py']
    sources=[ROOT/'scripts/coupled_contact_v13'/n for n in names];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    plan=dict(scope='Development-only clipping qualification before fresh inference/results',source_sha256=hashes,version=version,friction_cases=['edge_drop','squeeze'],sphere_cases=['curved','moving_sphere'],materials=CASES,cells=4,dt=.00005,friction_levels=[0,1,2],sphere_spacing_ladder_m=[[.0015,.00075],[.00075,.000375]],gates=dict(world_rms_m=.001,shape_rms_m=.0005,penetration_m=.001,temporal_world_rms_m=.00025,temporal_refinement_ratio=.8),acceptance='Friction level1 must agree with level2 for all4 cases; level0 diagnostic only. Sphere spacing accepted only if all4 cases pass. Accepted sphere spacing +friction1 then get3timelevels on all5dev0. No FEM spatial acceptance implied.')
    write(REPORT/f'clipped-qualification-plan-{version}.json',plan);_,compiler=load_kernel();rows={};checks=[]
    def obtain(specs):
        jobs=[s for s in specs if key(s) not in rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,a.workers)) as ex:
            for r in ex.map(task,[(s,version) for s in jobs]):rows[r['key']]=r
    specs=[(n,E,mu,5e-5,.0015,l) for n in plan['friction_cases'] for E,mu in CASES for l in [0,1,2]];obtain(specs)
    for n in plan['friction_cases']:
        for E,mu in CASES:
            aa=rows[key((n,E,mu,5e-5,.0015,1))];bb=rows[key((n,E,mu,5e-5,.0015,2))];c=compare(aa,bb)
            c.update(type='friction',scenario=n,young=E,friction=mu,levels=[1,2]);c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and physics(aa) and physics(bb);checks.append(c)
    write(REPORT/f'clipped-qualification-progress-{version}.json',dict(plan=plan,runs=list(rows.values()),checks=checks))
    spacing=None
    for coarse,fine in plan['sphere_spacing_ladder_m']:
        obtain([(n,E,mu,5e-5,h,1) for n in plan['sphere_cases'] for E,mu in CASES for h in [coarse,fine]]);stage=[]
        for n in plan['sphere_cases']:
            for E,mu in CASES:
                aa=rows[key((n,E,mu,5e-5,coarse,1))];bb=rows[key((n,E,mu,5e-5,fine,1))];c=compare(aa,bb)
                c.update(type='sphere_quadrature',scenario=n,young=E,friction=mu,spacings=[coarse,fine]);c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and physics(aa) and physics(bb);stage.append(c)
        checks.extend(stage);write(REPORT/f'clipped-qualification-progress-{version}.json',dict(plan=plan,runs=list(rows.values()),checks=checks))
        if all(c['passed'] for c in stage):spacing=coarse;break
    temporal=[]
    if spacing is not None and all(c['passed'] for c in checks if c['type']=='friction'):
        names=['squeeze','slide','moving_sphere','curved','edge_drop'];obtain([(n,6000.,.35,dt,spacing,1) for n in names for dt in [1e-4,5e-5,2.5e-5]])
        for n in names:
            aa,bb,cc=[rows[key((n,6000.,.35,dt,spacing,1))] for dt in [1e-4,5e-5,2.5e-5]];coarse=compare(aa,cc);c=compare(bb,cc)
            ratio=c.get('world_rms_m',float('inf'))/max(coarse.get('world_rms_m',0),1e-20)
            c.update(type='temporal',scenario=n,ratio=ratio);c['passed']=c['complete'] and c['world_rms_m']<=.00025 and ratio<=.8 and all(physics(r) for r in (aa,bb,cc));temporal.append(c)
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
    result=dict(plan=plan,compiler=compiler,runs=list(rows.values()),checks=checks,temporal_checks=temporal,accepted_sphere_spacing_m=spacing,friction_level=1,source_unchanged=unchanged,all_gates_pass=spacing is not None and bool(temporal) and all(c['passed'] for c in checks if c['type']=='friction') and all(c['passed'] for c in temporal) and unchanged)
    write(REPORT/f'clipped-qualification-{version}.json',result);write(REPORT/'clipped-development-qualification.json',result)
if __name__=='__main__':main()
