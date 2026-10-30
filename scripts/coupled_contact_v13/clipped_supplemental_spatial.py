"""Supplement spatial qualification to every development scene and rigid centers."""
from pathlib import Path
import sys, json, hashlib, time, concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.clipped_mesh_refinement import task,dense_compare
from scripts.coupled_contact_v13.surface_qualification import compare,key
from scripts.coupled_contact_v13.campaign import REPORT,write,remaining
CASES=[(n,E,mu) for n in ['squeeze','slide','moving_sphere'] for E,mu in [(6000.,.35),(3000.,.1)]]
def body(a,b):
    with np.load(ROOT/a['path']/'trajectory.npz') as aa,np.load(ROOT/b['path']/'trajectory.npz') as bb:
        x,y=aa['centers'],bb['centers']
        per_body=np.sqrt(np.mean((x[1:]-y[1:])**2,axis=(0,2)))
        return dict(finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),center_coordinate_rms_m=float(np.max(per_body)),per_body_coordinate_rms_m=per_body.tolist())
def main():
    declaration=dict(scope='Additional development-only spatial qualification; declared before inspecting these body outcomes',cases=CASES,soft_world_rms_m=.001,soft_shape_rms_m=.0005,body_center_coordinate_rms_m=.001,penetration_m=.001,minimum_J=.02,body_positions_finite=True,dt=.000025)
    write(REPORT/'clipped-supplemental-spatial-declaration.json',declaration)
    p=REPORT/'clipped-mesh-refinement.json'
    while not p.exists():
        if remaining()<=0:raise RuntimeError('Deadline')
        time.sleep(5)
    mainreport=json.loads(p.read_text())
    if not mainreport['all_gates_pass']:raise RuntimeError('Main spatial ladder did not qualify')
    low=mainreport['accepted_reference_cells'];high=low+2;spacing=mainreport['plan']['quadrature_spacing_m']
    sources=dict(mainreport['plan']['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for rel,h in sources.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+rel)
    version='supplement-'+hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
    specs=[(n,E,mu,cells,2.5e-5,spacing) for n,E,mu in CASES for cells in [low,high]]
    plan=dict(declaration=declaration,source_sha256=sources,version=version,cells=[low,high],main_report_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    write(REPORT/f'clipped-supplemental-spatial-plan-{version}.json',plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:rows=list(ex.map(task,[(s,version) for s in specs]))
    lookup={r['key']:r for r in rows};checks=[]
    for n,E,mu in CASES:
        a,b=[lookup[key((n,E,mu,c,2.5e-5,spacing))] for c in [low,high]]
        c=compare(a,b,True);c.update(scenario=n,young=E,friction=mu)
        c['dense_surface']=dense_compare(a,b)
        if a['success'] and b['success']:c['body']=body(a,b)
        d=c['dense_surface'];c['passed']=bool(c['complete'] and d['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and d['world_rms_m']<=.001 and d['shape_rms_m']<=.0005 and c['body']['finite'] and c['body']['center_coordinate_rms_m']<=.001 and all(r['min_j']>.02 and r['max_penetration_m']<=.001 for r in [a,b]));checks.append(c)
    unchanged=all(hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==h for r,h in sources.items())
    report=dict(plan=plan,runs=rows,checks=checks,source_unchanged=unchanged,all_gates_pass=unchanged and all(c['passed'] for c in checks))
    write(REPORT/'clipped-supplemental-spatial.json',report)
if __name__=='__main__':main()
