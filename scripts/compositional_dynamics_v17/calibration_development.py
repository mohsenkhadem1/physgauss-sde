"""Small-drop calibration/large-drop forecast data for matched-fit development.
Private simulator settings are not estimator inputs; no sealed-test claim.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,importlib.util,json,multiprocessing,shutil,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.materials import teacher_energy
PLAN=REPORT/'calibration-development-plan.json'
RAW=ROOT/'runs/compositional-dynamics-v17/calibration-development'
RESULT=REPORT/'calibration-development-results.json'


def prepare():
    if PLAN.exists() or RAW.exists():raise ValueError('Preserve previous calibration development')
    sources=closure([Path(__file__),ROOT/'scripts/compositional_dynamics_v17/prototype.py'])
    write(PLAN,dict(scope='Development knowngeometry/knownmass oracle surface-track calibration, not raw-video inference or qualified reference. Material labels/latent and future shape are private generator data, not estimator inputs.',
                    tasks=[dict(id='calibration',clearance_m=.01,duration_s=.20),dict(id='forecast',clearance_m=.04,duration_s=.25)],
                    n=3,dt_s=25e-6,gravity_m_s2=9.8,density_kg_m3=625.,dimensions_m=[.12,.06,.06],penalty_N_m3=4e7,
                    private_teacher_family='mooney_rivlin',private_teacher_latent=[.17,-.4],workers=2,seconds=900,
                    recorded_sample_interval_s=.001,observed_frame_interval_s=.01,
                    surface_query='Unique boundary of [0,.25,.5,.75,1]^3, np.unique lexicographic xyz:98points',
                    interior_query='[.25,.5,.75]^3:27points; knowninitial geometry only',
                    candidate_prior='Estimator must compare both MR/Yeoh familyflags and unknown continuousscale/shape from calibrationonly; no private latent/family input.',
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))


def verify(plan):
    if any(sha(ROOT/path)!=digest for path,digest in plan['source_sha256'].items()):raise ValueError('Frozen calibration source changed')


def worker(args):
    task,deadline=args;torch.set_num_threads(1);plan=json.loads(PLAN.read_text());verify(plan)
    folder=RAW/task['id'];folder.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,plan_sha256=sha(PLAN),path=str(folder.relative_to(ROOT)))
    try:
        spec=importlib.util.spec_from_file_location('private_calibration_generator',ROOT/'scripts/compositional_dynamics_v17/prototype.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
        impl.PLAN=PLAN;impl.RAW=folder/'work';constructor=impl.DynamicContactModel
        def configured(*args,**kwargs):
            kwargs['gravity']=(0,0,-plan['gravity_m_s2'])
            kwargs['bulk_energy']=lambda F,z:teacher_energy(F,z,plan['private_teacher_family'])
            kwargs['material_latent_field']=lambda q:q.new_tensor(plan['private_teacher_latent']).expand(*q.shape[:-1],2)
            return constructor(*args,**kwargs)
        impl.DynamicContactModel=configured
        settings=dict(dimensions=plan['dimensions_m'],young_Pa=12000.,poisson=.35,density=plan['density_kg_m3'],floor_z=-.03-task['clearance_m'],penalty_N_m3=plan['penalty_N_m3'],compression=0.,surface_order=8,sample_interval_s=plan['recorded_sample_interval_s'],duration=dict(drop=task['duration_s']))
        row=impl.run_case('drop',plan['n'],plan['dt_s'],settings,deadline)
        generated=impl.RAW/f"drop-n{plan['n']}-dt{plan['dt_s']:.8f}";shutil.copyfile(generated/'trajectory.npz',folder/'trajectory.npz')
        row['prototype_energy_scale_J']=row['energy_scale_J']
        mass=plan['density_kg_m3']*float(np.prod(plan['dimensions_m']))
        row['energy_scale_J']=max(abs(row['initial_energy_J']),mass*plan['gravity_m_s2']*task['clearance_m'],1e-10)
        with np.load(folder/'trajectory.npz') as a:row['maximum_relative_energy_drift']=float(np.max(abs(a['ledger'][:,1]-row['initial_energy_J']))/row['energy_scale_J'])
        row.update(identity);write(folder/'complete.json',row);return row
    except Exception as error:
        row=dict(**identity,completed=False,error=f'{type(error).__name__}: {error}');write(folder/'complete.json',row);return row


def run():
    plan=json.loads(PLAN.read_text());verify(plan)
    if RAW.exists():raise ValueError('Preserve existing development')
    RAW.mkdir(parents=True);start=time.monotonic();deadline=start+plan['seconds']-30
    report=dict(plan_sha256=sha(PLAN),rows=[],expected=2,terminal=False)
    pool=multiprocessing.get_context('spawn').Pool(plan['workers'])
    try:
        result=pool.imap_unordered(worker,[(task,deadline-10) for task in plan['tasks']])
        for _ in plan['tasks']:
            report['rows'].append(result.next(timeout=max(.01,deadline-time.monotonic())));write(RESULT,report)
    except Exception as error:report['error']=f'{type(error).__name__}: {error}'
    finally:
        pool.terminate();pool.join();rows=[];missing=[]
        for task in plan['tasks']:
            p=RAW/task['id']/'complete.json'
            if p.exists():
                row=json.loads(p.read_text())
                if row['plan_sha256']!=sha(PLAN) or row['task']!=task:raise ValueError('Identity mismatch')
                rows.append(row)
            else:missing.append(task['id'])
        verify(plan);report.update(rows=rows,missing=missing,terminal=True,completed=sum(r['completed'] for r in rows),seconds=time.monotonic()-start);write(RESULT,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else run()
