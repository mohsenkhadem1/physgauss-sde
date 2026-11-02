"""Development-only frozen bulk-law transfer to dynamics; known latent, no inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import importlib.util
import json
import multiprocessing
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT, REPORT, sha, write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.materials import teacher_energy
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK

PLAN=REPORT/'learned-dynamic-pilot-plan.json'
RAW=ROOT/'runs/compositional-dynamics-v17/learned-dynamic-pilot'
RESULT=REPORT/'learned-dynamic-pilot-results.json'


def prepare():
    if PLAN.exists() or RAW.exists():raise ValueError('Preserve previous pilot')
    tasks=[dict(case=case,family=family,method=method) for family in ('mooney_rivlin','yeoh') for case in ('release','drop') for method in ('teacher','frozen_shared')]
    for task in tasks:task['id']='-'.join(task[k] for k in ('family','case','method'))
    sources=closure([Path(__file__)])+[ROOT/'scripts/compositional_dynamics_v17/prototype.py']
    # Include the prototype dependency closure even though it is loaded privately.
    sources+=closure([ROOT/'scripts/compositional_dynamics_v17/prototype.py'])
    write(PLAN,dict(scope='Development known-latent constitutive transfer only. Both methods use same unqualified dynamic discretization; teacher trajectory is not independently validated truth. No heldout benchmark or inference claim.',
                    tasks=tasks,workers=2,seconds=1200,n=2,dt=25e-6,
                    settings=dict(dimensions=[.12,.06,.06],young_Pa=12000.,poisson=.35,density=625.,floor_z=-.07,penalty_N_m3=4e7,compression=.30,surface_order=8,sample_interval_s=.001,duration=dict(release=.05,drop=.25)),
                    material_latent=[0.,0.],family_flags=dict(mooney_rivlin=-1.,yeoh=1.),gravity_m_s2=9.81,
                    qualification='No dynamic reference qualification. Report all completion, minJ, penetration, energy and matched-time field errors descriptively. No result-based tuning or selecting best family.',
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))},
                    checkpoint_sha256={str(Path(BULK).relative_to(ROOT)):sha(BULK)}))


def verify(plan):
    for group in ('source_sha256','checkpoint_sha256'):
        for path,digest in plan[group].items():
            if sha(ROOT/path)!=digest:raise ValueError(f'Pinned input changed: {path}')


def worker(arguments):
    task,deadline=arguments;torch.set_num_threads(1)
    plan=json.loads(PLAN.read_text());verify(plan)
    destination=RAW/task['id'];destination.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,case=task['case'],family=task['family'],method=task['method'],plan_sha256=sha(PLAN),path=str(destination.relative_to(ROOT)))
    try:
        spec=importlib.util.spec_from_file_location('private_learned_pilot',ROOT/'scripts/compositional_dynamics_v17/prototype.py')
        implementation=importlib.util.module_from_spec(spec);spec.loader.exec_module(implementation)
        implementation.PLAN=PLAN;implementation.RAW=destination/'work';constructor=implementation.DynamicContactModel
        if task['method']=='frozen_shared':
            law=ScaleSeparatedMaterialLaw().double()
            law.load_state_dict(torch.load(BULK,map_location='cpu',weights_only=False)['state_dict']);law.eval()
            for parameter in law.parameters():parameter.requires_grad_(False)
            energy=law.energy
        else:
            energy=lambda F,z:teacher_energy(F,z[...,:2],task['family'])
        def configured(*args,**kwargs):
            kwargs['bulk_energy']=energy
            kwargs['material_latent_field']=lambda q:q.new_tensor([0.,0.,plan['family_flags'][task['family']]]).expand(*q.shape[:-1],3)
            return constructor(*args,**kwargs)
        implementation.DynamicContactModel=configured
        row=implementation.run_case(task['case'],plan['n'],plan['dt'],plan['settings'],deadline)
        generated=implementation.RAW/f"{task['case']}-n{plan['n']}-dt{plan['dt']:.8f}"
        shutil.copyfile(generated/'trajectory.npz',destination/'trajectory.npz')
        if task['case']=='drop' and 'trajectory_sha256' in row:
            # Prototype uses a fixed 2cm diagnostic scale; this pilot declares 4cm.
            row['prototype_energy_scale_J']=row['energy_scale_J']
            clearance=-plan['settings']['dimensions'][2]/2-plan['settings']['floor_z']
            mass=plan['settings']['density']*float(np.prod(plan['settings']['dimensions']))
            row['energy_scale_J']=max(abs(row['initial_energy_J']),mass*plan['gravity_m_s2']*clearance,1e-10)
            with np.load(destination/'trajectory.npz') as data:
                row['maximum_relative_energy_drift']=float(np.max(np.abs(data['ledger'][:,1]-row['initial_energy_J']))/row['energy_scale_J'])
        row.update(identity);write(destination/'complete.json',row);return row
    except Exception as error:
        row=dict(**identity,completed=False,error=f'{type(error).__name__}: {error}')
        write(destination/'complete.json',row);return row


def run():
    plan=json.loads(PLAN.read_text());verify(plan)
    if RAW.exists():raise ValueError('Preserve previous pilot')
    RAW.mkdir(parents=True);start=time.monotonic();deadline=start+plan['seconds']-30
    report=dict(plan_sha256=sha(PLAN),expected=len(plan['tasks']),rows=[],terminal=False)
    pool=multiprocessing.get_context('spawn').Pool(plan['workers'])
    try:
        results=pool.imap_unordered(worker,[(task,deadline-10) for task in plan['tasks']])
        for _ in plan['tasks']:
            report['rows'].append(results.next(timeout=max(.01,deadline-time.monotonic())));write(RESULT,report)
    except Exception as error:report['stage_error']=f'{type(error).__name__}: {error}'
    finally:
        pool.terminate();pool.join();rows=[];missing=[]
        for task in plan['tasks']:
            path=RAW/task['id']/'complete.json'
            if path.exists():
                row=json.loads(path.read_text())
                if row['plan_sha256']!=sha(PLAN) or row['task']!=task:raise ValueError('Pilot identity mismatch')
                rows.append(row)
            else:missing.append(task['id'])
        verify(plan);report.update(rows=rows,missing=missing,terminal=True,completed=sum(row['completed'] for row in rows),seconds=time.monotonic()-start)
        write(RESULT,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
