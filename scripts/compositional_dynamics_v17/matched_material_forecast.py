"""Forecast from committed inferred parameters and public commands only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,face_grid
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK
COMMIT=ROOT/'runs/compositional-dynamics-v17/matched-material-fit/fitted-parameter-commit.json'
PUBLIC=ROOT/'runs/compositional-dynamics-v17/calibration-public/metadata.json'
PLAN=REPORT/'matched-material-forecast-plan.json'
OUT=ROOT/'runs/compositional-dynamics-v17/matched-material-forecast'
FITPLAN=REPORT/'matched-material-fit-plan.json'
FITRESULT=REPORT/'matched-material-fit-results.json'


def configure(level):
    global COMMIT,PLAN,OUT,FITPLAN,FITRESULT
    suffix='-fine' if level=='fine' else ''
    COMMIT=ROOT/f'runs/compositional-dynamics-v17/matched-material{suffix}-fit/fitted-parameter-commit.json'
    PLAN=REPORT/f'matched-material{suffix}-forecast-plan.json'
    OUT=ROOT/f'runs/compositional-dynamics-v17/matched-material{suffix}-forecast'
    FITPLAN=REPORT/f'matched-material{suffix}-fit-plan.json'
    FITRESULT=REPORT/f'matched-material{suffix}-fit-results.json'


def prepare():
    if PLAN.exists() or OUT.exists():raise ValueError('Preserve forecast')
    fitted=json.loads(COMMIT.read_text());training=json.loads(FITPLAN.read_text());public=json.loads(PUBLIC.read_text())
    if not fitted['search_complete'] or fitted['plan_sha256']!=sha(FITPLAN) or fitted['results_sha256']!=sha(FITRESULT):raise ValueError('Completed committed fit required')
    sources=closure([Path(__file__),ROOT/'scripts/compositional_dynamics_v17/prototype.py'])
    write(PLAN,dict(scope='Development larger-drop prediction from fixed inferred parameters. No forecast response or private generator parameters read. Same learned model/discretization as calibration fit.',
                    task=fitted['task'],n=training['n'],dt_s=training['dt_s'],seconds=300,
                    geometry_dimensions_m=training['geometry_dimensions_m'],mass_kg=public['total_mass_kg'],setup=public['forecast_public_setup'],sample_interval_s=.001,
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
                    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [COMMIT,PUBLIC,Path(BULK),FITPLAN,FITRESULT]}))


def run():
    plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','input_sha256'):
        if any(sha(ROOT/path)!=digest for path,digest in plan[group].items()):raise ValueError('Frozen forecast input changed')
    if OUT.exists():raise ValueError('Preserve previous forecast')
    OUT.mkdir(parents=True);torch.set_num_threads(1)
    law=ScaleSeparatedMaterialLaw().double();law.load_state_dict(torch.load(BULK,map_location='cpu',weights_only=False)['state_dict']);law.eval()
    for parameter in law.parameters():parameter.requires_grad_(False)
    scale,shape,logpenalty=plan['task']['parameters'];flag=plan['task']['flag'];setup=plan['setup']
    spec=importlib.util.spec_from_file_location('private_committed_forecast',ROOT/'scripts/compositional_dynamics_v17/prototype.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
    impl.PLAN=PLAN;impl.RAW=OUT/'work';constructor=impl.DynamicContactModel
    def configured(*args,**kwargs):
        kwargs['gravity']=tuple(setup['gravity_vector_m_s2']);kwargs['bulk_energy']=law.energy
        kwargs['material_latent_field']=lambda q:q.new_tensor([scale,shape,flag]).expand(*q.shape[:-1],3)
        return constructor(*args,**kwargs)
    impl.DynamicContactModel=configured;dims=plan['geometry_dimensions_m']
    settings=dict(dimensions=dims,young_Pa=12000.,poisson=.35,density=plan['mass_kg']/np.prod(dims),floor_z=setup['floor_z_m'],penalty_N_m3=float(np.exp(logpenalty)),compression=0.,surface_order=8,sample_interval_s=plan['sample_interval_s'],duration=dict(drop=setup['duration_s']))
    row=impl.run_case('drop',plan['n'],plan['dt_s'],settings,time.monotonic()+plan['seconds']-15)
    source=impl.RAW/f"drop-n{plan['n']}-dt{plan['dt_s']:.8f}";shutil.copyfile(source/'trajectory.npz',OUT/'trajectory.npz')
    row['prototype_energy_scale_J']=row['energy_scale_J'];clearance=-dims[2]/2-setup['floor_z_m']
    row['energy_scale_J']=max(abs(row['initial_energy_J']),plan['mass_kg']*np.linalg.norm(setup['gravity_vector_m_s2'])*clearance,1e-10)
    public=json.loads(PUBLIC.read_text());q=np.asarray(public['material_queries']);dense=face_grid(17);point_indices=[np.flatnonzero(np.all(dense==point,axis=1))[0] for point in q]
    with np.load(OUT/'trajectory.npz') as data:
        row['maximum_relative_energy_drift']=float(np.max(abs(data['ledger'][:,1]-row['initial_energy_J']))/row['energy_scale_J'])
        indices=np.arange(0,len(data['times_s']),10);positions=data['surface_frames'][indices].reshape(len(indices),-1,3)[:,point_indices]
        np.savez_compressed(OUT/'surface-prediction.npz',times_s=data['times_s'][indices],positions_m=positions,material_queries=q)
    row.update(scope=plan['scope'],fitted_parameter_commit_sha256=sha(COMMIT),forecast_response_read=False,surface_prediction_sha256=sha(OUT/'surface-prediction.npz'))
    write(OUT/'complete.json',row)
    write(OUT/'prediction-commit.json',dict(plan_sha256=sha(PLAN),complete_sha256=sha(OUT/'complete.json'),surface_prediction_sha256=sha(OUT/'surface-prediction.npz'),trajectory_sha256=sha(OUT/'trajectory.npz'),completed=row['completed'],forecast_response_read=False))
    print(json.dumps(dict(completed=row['completed'],prediction=str(OUT/'surface-prediction.npz'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--level',choices=['coarse','fine'],default='coarse');a=p.parse_args();configure(a.level);prepare() if a.prepare else run()
