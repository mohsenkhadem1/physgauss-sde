"""Prospective numerical stress at parameter-grid endpoints; no reserved cases."""
from pathlib import Path
import sys,os,json,hashlib,time,concurrent.futures,multiprocessing
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scripts.coupled_contact_v13.surface_scenarios import run
from scripts.coupled_contact_v13.clipped_surface_contact import ClippedSurfaceFEMSystem
from scripts.coupled_contact_v13.campaign import remaining,write
BASE=ROOT/'runs/coupled-surface-v13/parameter-range-development'
REPORT=ROOT/'reports/coupled-surface-v13'

def task(job):
    name,E,mu=job;path=BASE/f'{name}-E{E:g}-mu{mu:g}';path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return json.loads((path/'complete.json').read_text())
    start=time.monotonic()
    try:
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached')
        class Contact(ClippedSurfaceFEMSystem):
            surface_spacing=.0015
            friction_quadrature_level=1
        result=run(name,E,mu,(4,4,4),.00005,system_class=Contact)
        np.savez_compressed(path/'trajectory.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        row={k:v for k,v in result.items() if isinstance(v,(float,int,str,bool))}
        row.update(success=True,finite=all(np.isfinite(result[k]).all() for k in ['frames','centers','reactions']))
        row['physics_gate_passed']=row['finite'] and row['min_j']>.02 and row['max_penetration_m']<=.001
    except Exception as e:row=dict(success=False,physics_gate_passed=False,error=repr(e))
    row.update(job=job,seconds=time.monotonic()-start,path=str(path.relative_to(ROOT)));write(path/'complete.json',row)
    print(json.dumps(row),flush=True);return row

def main():
    names=['physics.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','surface_scenarios.py','parameter_range_development.py']
    sources=[ROOT/'scripts/coupled_contact_v13'/n for n in names]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    jobs=[(s,E,mu) for s in ['squeeze','edge_drop'] for E in [2000.,18000.] for mu in [0.,.8]]
    plan=dict(scope='Prospective grid-endpoint numerical development, not fresh evaluation',jobs=jobs,cells=[4]*3,dt=.00005,sphere_spacing_m=.0015,friction_level=1,source_sha256=hashes,gates=dict(penetration_m=.001,min_j=.02,finite=True),interpretation='Stress screen only, not spatial/timestep qualification')
    write(REPORT/'parameter-range-development-plan.json',plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(task,jobs))
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in sources)
    write(REPORT/'parameter-range-development.json',dict(plan=plan,runs=rows,source_unchanged=unchanged,all_gates_pass=unchanged and all(r['physics_gate_passed'] for r in rows)))

if __name__=='__main__':main()
