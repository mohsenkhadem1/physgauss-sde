"""Predeclared development refinement ladder; never changes frozen forecasts."""
from pathlib import Path
import sys,os,json,hashlib,argparse,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scripts.coupled_contact_v13.surface_qualification import compare,key
from scripts.coupled_contact_v13.sparse_surface_contact import SparseSurfaceFEMSystem
from scripts.coupled_contact_v13.surface_scenarios import run,build
from scripts.coupled_contact_v13.surface_contact import box_surface_quadrature,tetrahedral_embedding
from scripts.coupled_contact_v13.mesh_diagnostic import shape,interpolation
from scripts.coupled_contact_v13.campaign import BASE,REPORT,write
CASES=[(n,E,mu) for n in ['edge_drop','curved'] for E,mu in [(6000.,.35),(3000.,.1)]]

def task(payload):
    spec,version=payload;name,E,mu,n,dt,spacing=spec
    path=BASE/'surface-qualification'/version/key(spec);path.mkdir(parents=True,exist_ok=True)
    if (path/'status.json').exists():return json.loads((path/'status.json').read_text())
    start=time.time()
    try:
        class Contact(SparseSurfaceFEMSystem):surface_spacing=spacing
        r=run(name,E,mu,(n,n,n),dt,system_class=Contact)
        query=build(name,E,mu,(2,2,2)).rest;W=interpolation(r['rest'],r['tets'],query)
        sampled=np.einsum('qn,tnc->tqc',W,r['frames'])
        np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},sampled=sampled)
        row={k:v for k,v in r.items() if isinstance(v,(str,float,int,bool))}
        row.update(success=True,finite_colliders=bool(np.isfinite(r['centers']).all() and np.isfinite(r['reactions']).all()),max_kabsch_deformation_m=max(shape(query,x) for x in sampled))
    except Exception as e:row=dict(success=False,error=repr(e))
    row.update(spec=list(spec),key=key(spec),path=str(path.relative_to(ROOT)),seconds=time.time()-start)
    write(path/'status.json',row);print(json.dumps(row),flush=True);return row

def dense_compare(a,b):
    if not a['success'] or not b['success']:return dict(complete=False)
    curves=[]
    for row in (a,b):
        with np.load(ROOT/row['path']/'trajectory.npz') as d:
            points,_=box_surface_quadrature(d['rest'],.003)
            ids,w=tetrahedral_embedding(d['rest'],d['tets'],points)
            curves.append(np.einsum('qi,tqic->tqc',w,d['frames'][:,ids]))
    x,y=curves
    return dict(complete=True,points=len(points),world_rms_m=float(np.sqrt(np.mean((x[1:]-y[1:])**2))),shape_rms_m=float(np.sqrt(np.mean([shape(i,j)**2 for i,j in zip(x[1:],y[1:])]))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-version',default='1f149be83bf2');p.add_argument('--workers',type=int,default=4);p.add_argument('--spacing',type=float,required=True);a=p.parse_args()
    sources=[ROOT/'scripts/coupled_contact_v13'/n for n in ['physics.py','surface_contact.py','sparse_surface_contact.py','surface_scenarios.py','surface_mesh_refinement.py','surface_qualification.py']]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};version='mesh-'+hashlib.sha256((json.dumps(hashes,sort_keys=True)+str(a.spacing)).encode()).hexdigest()[:12]
    plan=dict(scope='Development-only predeclared spatial refinement ladder, not held-out selection',source_sha256=hashes,version=version,prior_version=a.prior_version,ladder=[[6,8],[8,10]],cases=CASES,dt=.000025,quadrature_spacing_m=a.spacing,measurement='Both common27physicalrestpoints and common2678surfacequadraturepoints via actualtetP1interpolation, uniformcoordinateRMS, firstframeexcluded; BOTH mustpass samebounds',gates=dict(world_rms_m=.001,shape_rms_m=.0005,penetration_m=.001,temporal_world_rms_m=.00025),acceptance='Candidate lowerresolution only if all4spatialpairs and individualphysicsgates pass. If6vs8fails,8vs10 isnext. Acceptedcandidate gets base/halfdt on all5dev0 plusedge/curveddev1; otherwise explicitlyfailed.')
    pp=REPORT/f'surface-mesh-refinement-plan-{version}.json'
    if not pp.exists():write(pp,plan)
    prior=json.loads((REPORT/f'surface-qualification-plan-{a.prior_version}.json').read_text())
    for name in ['physics.py','surface_contact.py','surface_scenarios.py']:
        rel='scripts/coupled_contact_v13/'+name
        if prior['source_sha256'][rel]!=hashes[rel]:raise RuntimeError('Prior cache physics/scene changed: '+rel)
    rows={};checks=[];accepted=None
    def obtain(specs):
        missing=[]
        for spec in specs:
            k=key(spec)
            if k in rows:continue
            old=BASE/'surface-qualification'/a.prior_version/k/'status.json'
            if old.exists():rows[k]=json.loads(old.read_text())
            else:missing.append(spec)
        if missing:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,a.workers)) as ex:
                for r in ex.map(task,[(s,version) for s in missing]):rows[r['key']]=r
    def physics(r):return r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 and r['finite_colliders']
    for low,high in [(6,8),(8,10)]:
        specs=[(n,E,mu,cells,2.5e-5,a.spacing) for n,E,mu in CASES for cells in [low,high]];obtain(specs);stage=[]
        for n,E,mu in CASES:
            aa=rows[key((n,E,mu,low,2.5e-5,a.spacing))];bb=rows[key((n,E,mu,high,2.5e-5,a.spacing))]
            c=compare(aa,bb,True);c.update(scenario=n,young=E,friction=mu,cells=[low,high],type='spatial')
            dense=dense_compare(aa,bb);c['dense_surface']=dense
            c['passed']=c['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and dense['complete'] and dense['world_rms_m']<=.001 and dense['shape_rms_m']<=.0005 and physics(aa) and physics(bb);stage.append(c)
        checks.extend(stage);write(REPORT/f'surface-mesh-refinement-progress-{version}.json',dict(plan=plan,runs=list(rows.values()),checks=checks));print(json.dumps(stage),flush=True)
        if all(c['passed'] for c in stage):accepted=low;break
    temporal=[]
    if accepted is not None:
        cases=[(n,6000.,.35) for n in ['squeeze','slide','moving_sphere','edge_drop','curved']]+[(n,3000.,.1) for n in ['edge_drop','curved']]
        specs=[(n,E,mu,accepted,dt,a.spacing) for n,E,mu in cases for dt in [2.5e-5,1.25e-5]];obtain(specs)
        for n,E,mu in cases:
            aa=rows[key((n,E,mu,accepted,2.5e-5,a.spacing))];bb=rows[key((n,E,mu,accepted,1.25e-5,a.spacing))]
            c=compare(aa,bb);c.update(scenario=n,young=E,friction=mu,type='temporal',cells=accepted);c['passed']=c['complete'] and c['world_rms_m']<=.00025 and physics(aa) and physics(bb);temporal.append(c)
    oracle_bias=[]
    if accepted is not None and all(c['passed'] for c in temporal):
        specs=[(n,E,mu,cells,2.5e-5,a.spacing) for n,E,mu in CASES for cells in [2,3,4]];obtain(specs)
        for n,E,mu in CASES:
            ref=rows[key((n,E,mu,accepted,2.5e-5,a.spacing))]
            for cells in [2,3,4]:
                pred=rows[key((n,E,mu,cells,2.5e-5,a.spacing))];c=compare(pred,ref,True)
                c.update(scenario=n,young=E,friction=mu,predictor_cells=cells,reference_cells=accepted,dense_surface=dense_compare(pred,ref));oracle_bias.append(c)
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
    report=dict(plan=plan,runs=list(rows.values()),spatial_checks=checks,temporal_checks=temporal,accepted_reference_cells=accepted,oracle_predictor_bias=oracle_bias,source_unchanged=unchanged,all_gates_pass=accepted is not None and bool(temporal) and all(c['passed'] for c in temporal) and unchanged)
    write(REPORT/f'surface-mesh-refinement-{version}.json',report);write(REPORT/'surface-mesh-refinement.json',report)
if __name__=='__main__':main()
