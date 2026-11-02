"""Known-initial-geometry-only729point native capacity control; no additional tracks."""
import importlib.util,json,os,time,sys
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,AUTHOR,sha,write,KEYS
PUBLIC=ROOT/'runs/compositional-dynamics-v17/calibration-public'
DENSE=ROOT/'runs/compositional-dynamics-v17/native-phystwin-capacity-public'
FIT=ROOT/'runs/compositional-dynamics-v17/native-phystwin-capacity-fit'
PLAN=ROOT/'reports/compositional-dynamics-v17/native-phystwin-capacity-plan.json'
SOURCE=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_capacity_fit.py'

def geometry_only_auxiliary():
    coarse=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3)
    observed=np.unique(coarse[np.any((coarse==0)|(coarse==1),axis=1)],axis=0)
    fine=np.stack(np.meshgrid(*([np.linspace(0,1,9)]*3),indexing='ij'),-1).reshape(-1,3)
    boundary=fine[np.any((fine==0)|(fine==1),axis=1)]
    additional=np.asarray([x for x in boundary if not np.any(np.all(observed==x,axis=1))])
    interior=fine[np.all((fine>0)&(fine<1),axis=1)]
    assert observed.shape==(98,3)and additional.shape==(288,3)and interior.shape==(343,3)
    return observed,additional,interior

def densify_public(data,metadata):
    if set(data)!=KEYS:raise ValueError('Only original seven public arrays')
    observed,additional,interior=geometry_only_auxiliary();mapping=lambda q:q*[.12,.06,.06]+[0,-.03,-.03]
    if data['object_points'].shape!=(21,98,3)or not np.allclose(data['object_points'][0],mapping(observed),atol=1e-8,rtol=0):raise ValueError('Same canonical98observed correspondence')
    if not np.isclose(metadata['total_mass_kg'],.27,atol=1e-12,rtol=0):raise ValueError('Known0.27kg body')
    result={k:np.array(v,copy=True)for k,v in data.items()};result['surface_points']=mapping(additional).astype(np.float32);result['interior_points']=mapping(interior).astype(np.float32)
    meta=dict(metadata,object_masses_kg=[.27/729]*729,total_mass_kg=.27,interior_queries=interior.tolist(),auxiliary_surface_queries=additional.tolist(),native_mass_discretization='729equal known SI masses;98observed+288static auxiliary surface+343static interior. No new motion labels.',scope='Development geometry-only native capacity control; same98masked prefix tracks and known initial canonical geometry.')
    return result,meta

def prepare():
    if PLAN.exists()or DENSE.exists()or FIT.exists():raise ValueError('Preserve capacity preparation')
    with np.load(PUBLIC/'observations.npz')as f:data={k:f[k]for k in f.files}
    meta=json.loads((PUBLIC/'metadata.json').read_text());dense,metadata=densify_public(data,meta)
    DENSE.mkdir();np.savez_compressed(DENSE/'observations.npz',**dense);metadata['observations_sha256']=sha(DENSE/'observations.npz');write(DENSE/'metadata.json',metadata)
    prior=json.loads((ROOT/'reports/compositional-dynamics-v17/native-phystwin-fit-plan.json').read_text());sources=dict(prior['source_sha256'])
    for path in [Path(__file__),SOURCE,ROOT/'scripts/compositional_dynamics_v17/native_phystwin_replay.py',ROOT/'scripts/compositional_dynamics_v17/native_phystwin_graph_recovery.py',ROOT/'tests/test_native_phystwin_capacity_v17.py']:sources[str(path.relative_to(ROOT))]=sha(path)
    inputs=[PUBLIC/'observations.npz',PUBLIC/'metadata.json',DENSE/'observations.npz',DENSE/'metadata.json',ROOT/'runs/compositional-dynamics-v17/render-environment-freeze.txt']
    setup=dict(duration_s=.20,frame_dt_s=.01,floor_z_m=meta['public_floor_z_m'],initial_velocity_m_s=[0,0,0],initial_geometry_same=True,gravity_vector_m_s2=[0,0,-9.8])
    write(PLAN,dict(scope='Development native capacity control before any held target scoring;125particle result retained. Choose futurecohort baseline capacity using CALIBRATION only.',particles=729,observed_tracks=98,static_auxiliary_surface=288,static_interior=343,initial_graph_radius_m=.02,graph_radius_search_m=[.01,.05],initial_max_neighbors=30,mass_kg=.27,cma_generations=20,epochs=200,seed=42,dt_s=.000025,substeps=400,source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs},calibration_setup=setup,forecast_setup=meta['forecast_public_setup'],stage_seconds=3600,physical_gpu=0,KillMode='control-group',method_change='Only known initial spatial discretization and corresponding SI particle mass/coefficient scaling; author algorithm/loss/budgets unchanged. Native initial radius.02 is real.yaml default and appropriate q9spacing. No extra observed deformation labels.'))

def guard():
    p=json.loads(PLAN.read_text())
    for group in ['source_sha256','input_sha256']:
        for path,digest in p[group].items():
            if sha(ROOT/path)!=digest:raise ValueError('Capacity source/input changed '+path)
    return p

def run():
    p=guard();review=ROOT/'reports/compositional-dynamics-v17/native-phystwin-capacity-prefit-independent-review.json';audit=json.loads(review.read_text())
    if audit['passed']is not True or audit['plan_sha256']!=sha(PLAN):raise ValueError('Exact capacity prefit review required')
    marker=DENSE/'stage-start.json'
    with marker.open('x')as f:json.dump(dict(plan_sha256=sha(PLAN),started_unix=time.time(),stage_seconds=3600),f)
    start=time.monotonic();status=dict(completed=False,plan_sha256=sha(PLAN))
    try:
        spec=importlib.util.spec_from_file_location('_native_capacity_fit',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.run('fit')
        replay_source=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_replay.py';spec=importlib.util.spec_from_file_location('_native_capacity_replay',replay_source);rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep);rep.SOURCE=SOURCE
        for name,setup in [('calibration',p['calibration_setup']),('forecast',p['forecast_setup'])]:
            if time.monotonic()-start>p['stage_seconds']-30:raise TimeoutError('Shared stage deadline')
            out=ROOT/f'runs/compositional-dynamics-v17/native-phystwin-capacity-{name}'
            rep.replay(FIT,DENSE,setup,out,sha(PLAN))
        guard();status['completed']=True
    except Exception as exc:status['error']=repr(exc);raise
    finally:status['wall_seconds']=time.monotonic()-start;write(DENSE/'stage-complete.json',status)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);s=a.parse_args();globals()[s.stage]()
