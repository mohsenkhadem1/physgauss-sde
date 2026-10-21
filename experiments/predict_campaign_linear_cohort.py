"""All declared supplied-physics linear predictions, with no finite labels.

Each physical hypothesis constructs one rest tangent/mass model and predicts
each of its two predeclared commands once from rest. No fitting or retries.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).resolve()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def resolve(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=path:raise ValueError('Repository-relative path required')
    result=(ROOT/p).resolve()
    if not result.is_relative_to(ROOT):raise ValueError('Path escapes repository')
    return result


def check(protocol):
    if protocol['schema']!='linear-known-hypothesis-cohort-v0.1' or protocol['private_hz']!=120 or protocol['workers']!=4:
        raise ValueError('Exactly predeclared all96 common120Hz/four-worker scope required')
    for path,digest in protocol['source_sha256'].items():
        if sha(resolve(path))!=digest:raise ValueError('Frozen source changed '+path)
    source=protocol['input_design']
    if sha(resolve(source['path']))!=source['sha256']:raise ValueError('Finite hypothesis design changed')
    design=json.loads(resolve(source['path']).read_text())
    if len(design['hypotheses'])!=48 or len(design['cells'])!=96 or len({c['id'] for c in design['cells']})!=96:
        raise ValueError('Exact48 hypotheses/96 unique commands required')
    if any(c['split']!='train' for c in design['cells']):raise ValueError('Only declared TRAIN inputs permitted')
    return design


def mechanics(result,hardware,spec):
    q=result['positions_m'].numpy();x=hardware.points;tets=hardware.tets
    dm=np.swapaxes(x[tets[:,1:]]-x[tets[:,:1]],1,2)
    ds=np.swapaxes(q[:,tets[:,1:]]-q[:,tets[:,:1]],2,3)
    J=np.linalg.det(ds@np.linalg.inv(dm)[None])
    linear_error=0.;distance_error=0.
    for j,h in enumerate(hardware.handles):
        local=x[h.ordered_patch_nodes]-x[h.origin_node_id]
        omega=result['handle_rotation_coordinates_rad'][:,j].numpy()
        expected=result['handle_centers_m'][:,j,None].numpy()+local[None]+np.cross(omega[:,None],local[None])
        linear_error=max(linear_error,float(np.max(abs(q[:,h.ordered_patch_nodes]-expected))))
        actual=q[:,h.ordered_patch_nodes];old=np.linalg.norm(local[:,None]-local[None,:],axis=-1)
        current=np.linalg.norm(actual[:,:,None]-actual[:,None,:],axis=-1)
        distance_error=max(distance_error,float(np.max(abs(current-old))))
    return dict(body_mass_kg=float(spec['material']['density_kg_m3']*np.sum(abs(np.linalg.det(dm))/6)),
        handle_mass_total_kg=float(sum(h.mass_kg for h in hardware.handles)),
        minimum_saved_J=float(J.min()),maximum_saved_J=float(J.max()),
        positive_saved_J=bool(np.all(J>0)),minimum_saved_J_is_sampled_only=True,
        maximum_linear_patch_map_defect_m=linear_error,maximum_finite_rigid_patch_distance_defect_m=distance_error,
        maximum_nodal_displacement_m=float(np.linalg.norm(q-x,axis=-1).max()),
        rotation_semantics='infinitesimal world rotation coordinates; not finite SO3')


def predict_hypothesis(protocol,hypothesis_id):
    import torch
    try:
        import campaign_hypothesis_inputs as inputs
        import campaign_exciting_commands as commands
        from campaign_linear_hypothesis import LinearHypothesisDynamics
    except ImportError:
        from experiments import campaign_hypothesis_inputs as inputs
        from experiments import campaign_exciting_commands as commands
        from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
    design=check(protocol);hyp=next(h for h in design['hypotheses'] if h['id']==hypothesis_id)
    cells=[c for c in design['cells'] if c['hypothesis_id']==hypothesis_id]
    public=next(h for h in design['public_hardware'] if h['id']==hyp['public_hardware_id'])
    hardware=inputs.known_hardware_from_public(public);spec=hyp['spec'];rows=[]
    model=None;construction_error=None;started=time.monotonic()
    try:
        # Preserve the immutable constructor's ordinary2-thread assembly.
        model=LinearHypothesisDynamics.from_hypothesis(hardware,spec)
    except Exception as error:construction_error=f'{type(error).__name__}: {error}'
    torch.set_num_threads(1)
    for cell in cells:
        directory=resolve(protocol['output'])/'cells'/cell['id'];directory.mkdir(parents=True,exist_ok=False)
        row=dict(cell=cell,passed=False,pid=os.getpid(),constructor_threads=2,solve_threads=torch.get_num_threads(),
            supplied_physics=True,posterior=False,finite_response_inputs_used=False)
        begin=time.monotonic()
        try:
            if construction_error:raise ValueError(construction_error)
            clock=commands.make_exciting_commands(cell['program_id'])
            result=model.predict(clock,integration_hz=120)
            for key in ['positions_m','velocities_point_chord_m_s','handle_centers_m','handle_rotation_coordinates_rad','actual_state_times_s']:
                if not bool(torch.isfinite(result[key]).all()):raise ValueError('Nonfinite prediction '+key)
            if tuple(result['positions_m'].shape)!=(61,len(hardware.points),3):raise ValueError('Incomplete prediction')
            if np.max(abs(result['actual_state_times_s'].numpy()-clock.output_times_s))>1e-12:raise ValueError('Prediction state clock mismatch')
            destination=directory/'predictions.npz'
            np.savez_compressed(destination,positions_m=result['positions_m'].numpy(),
                velocities_point_chord_m_s=result['velocities_point_chord_m_s'].numpy(),
                handle_centers_m=result['handle_centers_m'].numpy(),
                handle_rotation_coordinates_rad=result['handle_rotation_coordinates_rad'].numpy(),
                actual_state_times_s=result['actual_state_times_s'].numpy(),times_s=clock.output_times_s,
                id=np.array(cell['id']),evaluation_role=np.array(cell['evaluation_role']))
            row.update(passed=True,prediction={'path':str(destination.relative_to(ROOT)),'sha256':sha(destination)},
                mechanics=mechanics(result,hardware,spec),public_command_manifest=clock.public_manifest(),
                physical_factorizations=model.factorizations)
        except Exception as error:row['error']=f'{type(error).__name__}: {error}'
        row['elapsed_s']=time.monotonic()-begin;write(directory/'summary.json',row);rows.append(row)
    return dict(hypothesis_id=hypothesis_id,rows=rows,elapsed_s=time.monotonic()-started)


def run(protocol_path,digest):
    if sha(resolve(protocol_path))!=digest:raise ValueError('Prediction protocol changed')
    protocol=json.loads(resolve(protocol_path).read_text());design=check(protocol);out=resolve(protocol['output'])
    out.mkdir(parents=True,exist_ok=False);(out/'cells').mkdir();write(out/'protocol.json',protocol)
    status=dict(terminal=False,completed=False,pid=os.getpid(),planned_cells=96,completed_cells=0,failed_cells=0,
        workers=4,ordinary_constructor_threads_each=2,linear_solve_threads_each=1,gpu_used=False)
    write(out/'execution.json',status);rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(predict_hypothesis,protocol,h['id']):h['id'] for h in design['hypotheses']}
        for future in as_completed(futures):
            hypothesis=futures[future]
            try:result=future.result();new=result['rows']
            except Exception as error:
                new=[dict(cell=c,passed=False,error=f'Worker failure {type(error).__name__}: {error}') for c in design['cells'] if c['hypothesis_id']==hypothesis]
            rows.extend(new);status.update(completed_cells=len(rows),failed_cells=sum(not r['passed'] for r in rows),elapsed_s=time.monotonic()-started)
            write(out/'execution.json',status);write(out/'progress.json',rows);print(json.dumps(status),flush=True)
    check(protocol)
    order={c['id']:i for i,c in enumerate(design['cells'])};rows.sort(key=lambda r:order[r['cell']['id']])
    all_pass=len(rows)==96 and all(r['passed'] for r in rows)
    if all_pass:
        arrays={k:[] for k in ['positions_m','velocities_point_chord_m_s','handle_centers_m','handle_rotation_coordinates_rad']}
        for row in rows:
            with np.load(resolve(row['prediction']['path']),allow_pickle=False) as d:
                for key in arrays:arrays[key].append(d[key])
        np.savez_compressed(out/'predictions.npz',**{k:np.stack(v) for k,v in arrays.items()},
            ids=np.array([r['cell']['id'] for r in rows]),times_s=np.arange(61)/20,
            evaluation_roles=np.array([r['cell']['evaluation_role'] for r in rows]))
    write(out/'summary.json',dict(completed=all_pass,declared_cells=96,rows=rows,
        cohort_predictions_sha256=sha(out/'predictions.npz') if all_pass else None,
        role='supplied_physics_unfitted_small_strain_baseline_not_posterior',finite_labels_used=False,
        cohort_finite_comparison_pending=True,cohort_score_available=False,production_dataset_qualified=False,
        limitation='Finite rigid patches become infinitesimal rotations; sampled J and patch-distance defects are diagnostics, never selection filters. All96 cells are required for a cohort score.'))
    status.update(terminal=True,completed=all_pass,elapsed_s=time.monotonic()-started);write(out/'execution.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args();run(a.protocol,a.protocol_sha256)
