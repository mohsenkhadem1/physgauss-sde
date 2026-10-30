"""Development-only consistent-mass native spring timestep diagnostic, engineered scenes.

Two predeclared spring scales, squeeze and curved push, base and half time step.
No reserved observations/outcomes and no fitted test parameters are consumed.
Explicit settings are required so an unqualified default cannot silently run.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scripts.coupled_contact_v13.campaign import remaining,write
BASE=ROOT/'runs/coupled-surface-v13/development/native-timestep'
REPORT=ROOT/'reports/coupled-surface-v13'


def job(spec):
    from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem
    from scripts.coupled_contact_v13.clipped_surface_contact import with_clipped_surface_contact
    from scripts.coupled_contact_v13.engineered_scenarios import build,command
    from scripts.coupled_contact_v13.consistent_mass import with_consistent_mass
    started=time.monotonic()
    cls=type('DevelopmentClippedNative',(with_consistent_mass(with_clipped_surface_contact(NativeSpringSystem)),),
             dict(friction_quadrature_level=spec['friction_level'],surface_spacing=spec['sphere_spacing']))
    system=build(spec['scene'],young=spec['scale'],friction=spec['friction'],cells=(spec['cells'],)*3,system_class=cls)
    duration=.6;steps=int(np.ceil(duration/spec['dt']));dt=duration/steps
    frames=[system.x.copy()];times=[0.];next_save=1;error=None;last_time=0.;last_j=1.
    try:
        for i in range(steps):
            if i%100==0 and remaining()<=0:raise RuntimeError('Original campaign deadline reached')
            system.step(dt,command(spec['scene'],i*dt));last_time=(i+1)*dt
            last_j=float(np.linalg.det(system.deformation()).min())
            if next_save<=60 and last_time>=next_save/100-1e-12:
                frames.append(system.x.copy());times.append(last_time);next_save+=1
    except Exception as exc:error=repr(exc)
    path=BASE/spec['tag']/spec['name'];path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'trajectory.npz',frames=np.asarray(frames),times=np.asarray(times),rest=system.rest,tets=system.tets)
    result=dict(spec=spec,success=error is None and len(frames)==61,error=error,
                last_successful_time_s=last_time,last_successful_min_j=last_j,
                stopped_state_min_j=float(np.linalg.det(system.deformation()).min()),
                integrated_min_j=system.min_j,max_penetration_m=system.max_penetration,
                saved_frames=len(frames),elapsed_seconds=time.monotonic()-started,
                native_provenance=system.metadata(),contact_class='Consistent P1 inertia, clipped native, engineered scenes; prospective development candidate')
    write(path/'diagnostic.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cells',type=int,required=True);parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--friction-level',type=int,required=True);parser.add_argument('--sphere-spacing',type=float,required=True)
    parser.add_argument('--friction',type=float,default=.6);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--tag',required=True);args=parser.parse_args()
    if args.cells<1 or args.dt<=0 or args.sphere_spacing<=0 or args.workers<1:raise ValueError('Positive discretization/worker values required')
    if '/' in args.tag or '..' in args.tag:raise ValueError('Tag must be a simple name')
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Development diagnostic must be declared before fresh freeze')
    specs=[dict(cells=args.cells,dt=args.dt/factor,friction_level=args.friction_level,sphere_spacing=args.sphere_spacing,
                friction=args.friction,scale=scale,scene=scene,tag=args.tag,name=f'{scene}-{int(scale)}-{factor}')
           for scene in ('squeeze','curved') for scale in (1e5,1e6) for factor in (1,2)]
    REPORT.mkdir(parents=True,exist_ok=True)
    sources=['native_consistent_development.py','native_adapter.py','physics.py','engineered_scenarios.py','consistent_mass.py',
             'surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp']
    hashes={str((ROOT/'scripts/coupled_contact_v13'/name).relative_to(ROOT)):hashlib.sha256((ROOT/'scripts/coupled_contact_v13'/name).read_bytes()).hexdigest() for name in sources}
    write(REPORT/f'native-timestep-{args.tag}-declaration.json',dict(settings=vars(args),jobs=specs,source_sha256=hashes,scope=__doc__))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(job,specs))
    comparisons=[]
    for scene in ('squeeze','curved'):
        for scale in (1e5,1e6):
            pair=[r for r in rows if r['spec']['scene']==scene and r['spec']['scale']==scale]
            a,b=pair;comparison=dict(scene=scene,scale=scale,base_success=a['success'],half_success=b['success'],base_failure_time=a['last_successful_time_s'],half_failure_time=b['last_successful_time_s'])
            if a['success'] and b['success']:
                with np.load(BASE/args.tag/a['spec']['name']/'trajectory.npz') as aa,np.load(BASE/args.tag/b['spec']['name']/'trajectory.npz') as bb:
                    if not np.allclose(aa['times'],bb['times'],atol=1e-10,rtol=0):raise ValueError('Diagnostic saved time grids differ')
                    comparison['trajectory_coordinate_rms_m']=float(np.sqrt(np.mean((aa['frames']-bb['frames'])**2)))
            comparisons.append(comparison)
    write(REPORT/f'native-timestep-{args.tag}.json',dict(settings=vars(args),source_sha256=hashes,rows=rows,comparisons=comparisons,
          interpretation='Halving dt can reveal step-size instability but agreement/failure at two levels does not establish continuum convergence or identify physical collapse. Any common-step revision must be declared before fresh freeze.'))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
