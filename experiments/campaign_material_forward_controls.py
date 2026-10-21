"""Fresh true-material forward controls, unlocked only by frozen inference.

No fitting, source changes, posterior/reference reruns, or finite dynamics.
Truth and held motion are opened only AFTER marker and source verification.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import time
import numpy as np
import torch
try:
    from campaign_hypothesis_inputs import known_hardware_from_public,validate_realized_hypothesis
    from campaign_linear_hypothesis import LinearHypothesisDynamics
    from campaign_modal_hypothesis import FixedModalLinearDynamics
    from campaign_modal_static_lift import complete_static_lift
    from campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
    import campaign_authored_force_audit as pure
except ImportError:
    from experiments.campaign_hypothesis_inputs import known_hardware_from_public,validate_realized_hypothesis
    from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
    from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
    from experiments.campaign_modal_static_lift import complete_static_lift
    from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
    from experiments import campaign_authored_force_audit as pure

ROOT=Path(__file__).resolve().parents[1]
DESIGN_SHA='5a37656704b1cbd4a18477126be0f295d036a582eaa0130d6e0484e9da8e2e46'
DESIGN='runs/operator-dev/material-oracle-depth-v01/protocol-design.json'
MARKER='runs/gaussian-campaign-v01/material-oracle-depth-v01/importance-reference-v01/posterior-frozen.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=path:raise ValueError('Canonical repository-relative path required')
    result=(ROOT/p).resolve()
    if not result.is_relative_to(ROOT):raise ValueError('Repository escape')
    return result


def checked(record):
    p=resolve(record['path'])
    if sha(p)!=record['sha256']:raise ValueError('Pinned input changed '+record['path'])
    return p


def patch_defect(q,hardware):
    defect=0.
    for h in hardware.handles:
        ids=h.ordered_patch_nodes;actual=q[:,ids];rest=hardware.points[ids]
        expected=np.linalg.norm(rest[:,None]-rest[None,:],axis=-1)
        distances=np.linalg.norm(actual[:,:,None]-actual[:,None,:],axis=-1)
        defect=max(defect,float(np.max(abs(distances-expected))))
    return defect


def comparison(predicted,reference):
    error=predicted-reference
    return {'coordinate_RMSE_m':float(np.sqrt(np.mean(error**2))),
        'nodal_vector_RMSE_m':float(np.sqrt(np.mean(np.sum(error**2,axis=-1)))),
        'maximum_nodal_vector_error_m':float(np.linalg.norm(error,axis=-1).max())}


def run(output):
    # This is deliberately the FIRST input read. Do not move any truth/design
    # or held-response read above the frozen-posterior verification block.
    marker_path=resolve(MARKER)
    if not marker_path.exists():raise ValueError('Posterior-frozen marker absent; truth remains unopened')
    marker=read(marker_path)
    if not (marker['held_motion_opened'] is False and marker['material_truth_opened'] is False
            and marker['inference_completed_without_truth_labels'] is True):raise ValueError('Inference ordering marker invalid')
    summaries=marker_path.parent/'reference-summaries.json'
    if sha(summaries)!=marker['reference_summaries_sha256']:raise ValueError('Frozen posterior summaries changed')
    for name,digest in marker['source_sha256'].items():
        if sha(resolve(name))!=digest:raise ValueError('Frozen inference source changed')
    if sha(resolve(DESIGN))!=DESIGN_SHA:raise ValueError('Prospective design changed')
    design=read(resolve(DESIGN));public=read(checked(design['public_hardware']))
    # First truth read occurs only here, after successful marker verification.
    spec=read(checked(design['truth_spec']));known=known_hardware_from_public(public)
    validate_realized_hypothesis(known,spec)
    protocols=[read(checked(record)) for record in design['physics_cases']]
    if {p['program_id'] for p in protocols}!=set(PROGRAM_IDS):raise ValueError('Exactly predeclared A/B controls required')
    out=resolve(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.monotonic()
    full=LinearHypothesisDynamics.from_hypothesis(known,spec)
    modal=complete_static_lift(FixedModalLinearDynamics(full,96));rows=[]
    oldspec={**spec,'load_nodes':[h.origin_node_id for h in known.handles],
        'load_patches':[h.ordered_patch_nodes.tolist() for h in known.handles]}
    oldhardware=[dict(mass_kg=h.mass_kg,full_COM_inertia_kg_m2=h.com_inertia_body_kg_m2,COM_offset_m=h.com_offset_body_m) for h in known.handles]
    for protocol in protocols:
        program=protocol['program_id'];directory=resolve(protocol['output'])
        execution=read(directory/'execution.json');summary=read(directory/'summary.json')
        if not (execution['terminal'] and execution['completed'] and summary['completed']
                and not Path('/proc',str(execution['pid'])).exists()):raise ValueError('Finite worker incomplete/live')
        if sha(directory/'trajectory.npz')!=summary['trajectory_sha256']:raise ValueError('Finite response hash changed')
        if read(directory/'private-provenance/realized-hypothesis.json')!=spec or read(directory/'public-hardware.json')!=public:raise ValueError('Finite truth/hardware differs')
        clock=make_exciting_commands(program)
        if read(directory/'public-command-manifest.json')!=clock.public_manifest():raise ValueError('Finite known commands differ')
        finite=pure.archive(directory/'trajectory.npz')
        pure.exact(finite['times_s'],clock.output_times_s,'Finite output clock differs')
        pure.close(finite['actual_state_times_s'],clock.output_times_s,1e-12,'Finite actual clock differs')
        geometry,_,_,_=pure.frame_mechanics(finite,known.points,known.tets,oldspec,oldhardware,spec['material'])
        predictions={'full_linear':full.predict(clock,integration_hz=120),
            'enriched_modal99':modal.predict(clock,spec['material'],integration_hz=120)}
        row={'program':program,'finite_trajectory_sha256':sha(directory/'trajectory.npz'),
            'finite_geometry':geometry,'finite_patch_distance_defect_m':patch_defect(finite['positions_m'],known),'controls':{}}
        for name,result in predictions.items():
            arrays={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in result.items() if isinstance(v,(torch.Tensor,np.ndarray))}
            q=arrays['positions_m'];v=arrays['velocities_point_chord_m_s']
            pure.close(arrays['actual_state_times_s'],clock.output_times_s,1e-12,'Control actual clock differs')
            destination=out/(program+'__'+name+'.npz');np.savez_compressed(destination,**arrays)
            metrics=comparison(q,finite['positions_m']);metrics['velocity_coordinate_RMSE_m_s']=float(np.sqrt(np.mean((v-finite['velocities_point_chord_m_s'])**2)))
            metrics['finite_rigid_patch_distance_defect_m']=patch_defect(q,known)
            metrics['handle_center_coordinate_RMSE_m']=float(np.sqrt(np.mean((arrays['handle_centers_m']-finite['handle_centers_m'])**2)))
            metrics['prediction_sha256']=sha(destination);row['controls'][name]=metrics
        a=predictions['full_linear'];b=predictions['enriched_modal99']
        row['modal_vs_full_linear']=comparison(b['positions_m'].numpy(),a['positions_m'].numpy())
        row['modal_vs_full_linear']['velocity_coordinate_RMSE_m_s']=float(torch.sqrt(torch.mean((b['velocities_point_chord_m_s']-a['velocities_point_chord_m_s'])**2)))
        rows.append(row);print(json.dumps(row),flush=True)
    report={'status':'COMPLETED_UNFITTED_KNOWN_TRUE_MATERIAL_FORWARD_CONTROLS','rows':rows,'elapsed_s':time.monotonic()-started,
        'truth_read_only_after_posterior_frozen_verification':True,'frozen_marker_sha256':sha(marker_path),
        'reference_summaries_sha256':sha(summaries),'prospective_design_sha256':DESIGN_SHA,
        'source_sha256':sha(__file__),'frozen_inference_source_sha256':marker['source_sha256'],
        'basis':modal.basis_diagnostics,'CPU_threads':2,'GPU_used':False,'predictions_per_model_per_program':1,
        'fit_or_tuning_performed':False,'posterior_or_importance_reference_rerun':False,'new_finite_runs':0,
        'limitation':'Supplied true material separates linear/model-reduction forward error from inference error. Finite reference remains independent discretized mechanics, not native Isaac or continuum truth. No calibration claim.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();run(args.output)
