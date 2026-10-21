"""Four supplied-truth forward controls, locked until joint posterior freeze.

No finite mechanics, cache rebuilding, fitting, inference or reference reruns.
Use the exact inference cache on CPU, not a freshly reconstructed modal basis.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
try:
    from campaign_material_forward_controls import ROOT,resolve,read,checked,sha,patch_defect,comparison
    from campaign_hypothesis_inputs import known_hardware_from_public,validate_realized_hypothesis
    from campaign_linear_hypothesis import LinearHypothesisDynamics
    from campaign_support_basis_cache import load_support_basis_cache
    from campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
    import campaign_authored_force_audit as pure
except ImportError:
    from experiments.campaign_material_forward_controls import ROOT,resolve,read,checked,sha,patch_defect,comparison
    from experiments.campaign_hypothesis_inputs import known_hardware_from_public,validate_realized_hypothesis
    from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
    from experiments.campaign_support_basis_cache import load_support_basis_cache
    from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
    from experiments import campaign_authored_force_audit as pure

MARKER='runs/gaussian-campaign-v01/unknown-point-oracle-v01/joint-reference-v01/posterior-frozen.json'
DESIGN='runs/operator-dev/unknown-point-oracle-inputs-v01/protocol-design.json'
DESIGN_SHA='f50c2de50217904b215df5e25d93b6699c19b27e48941f21233a1a1a36be9fe9'
CACHE='runs/gaussian-campaign-v01/point-support-bases-v01/cache-manifest.json'
CACHE_SHA='642f50c696129c0cafda781b87a3342d3effcc25ca5484172afb25b604977664'


def frozen_gate():
    """No design, physical-truth or held-response access before this returns."""
    marker_path=resolve(MARKER)
    if not marker_path.exists():raise ValueError('Joint posterior-frozen marker absent; truth/held motion remain unopened')
    marker=read(marker_path)
    if (marker.get('physical_truth_opened') is not False
            or marker.get('held_motion_used_for_inference') is not False
            or marker.get('held_motion_evaluated') is not False):
        raise ValueError('Joint inference ordering marker invalid')
    reference=marker_path.parent
    if sha(reference/'reference-summaries.json')!=marker['reference_summaries_sha256']:
        raise ValueError('Frozen joint reference summaries changed')
    records=marker['references']
    if len(records)!=3 or {Path(r['path']).name for r in records}!={f'reference-{seed}.npz' for seed in [20261112,20261113,20261114]}:
        raise ValueError('All3 prospectively fixed reference-particle archives required')
    for record in records:checked(record)
    for path,digest in marker['source_sha256'].items():
        if sha(resolve(path))!=digest:raise ValueError('Frozen joint source changed '+path)
    protocol=read(reference/'protocol.json')
    if (protocol['design_path']!=DESIGN or protocol['design_sha256']!=DESIGN_SHA
            or protocol['cache_path']!=CACHE or protocol['cache_sha256']!=CACHE_SHA):
        raise ValueError('Forward controls must use exact frozen inference design/cache')
    if sha(resolve(DESIGN))!=DESIGN_SHA:raise ValueError('Prospective input design changed')
    return marker,marker_path


def run(output):
    # FIRST input read is the metadata gate. No truth/held file above this line.
    marker,marker_path=frozen_gate()
    design=read(resolve(DESIGN));public=read(checked(design['public_hardware']))
    # Truth is read only now, once all joint-reference hashes are verified.
    spec=read(checked(design['truth_spec']));known=known_hardware_from_public(public)
    validate_realized_hypothesis(known,spec)
    protocols=[read(checked(record)) for record in design['physics_cases']]
    if {p['program_id'] for p in protocols}!=set(PROGRAM_IDS):raise ValueError('Exactly the frozen A/B cases required')
    torch.set_num_threads(2);out=resolve(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    packed,cache_public,ids,_=load_support_basis_cache(CACHE,CACHE_SHA)
    if cache_public!=public or len(ids)!=55:raise ValueError('Exact public hardware/all55 categories required')
    node=spec['fixture_nodes'][0]
    if node not in ids:raise ValueError('True support outside normalized catalogue; no fallback or reconstruction')
    full=LinearHypothesisDynamics.from_hypothesis(known,spec)
    theta=torch.tensor([[spec['material'][k] for k in ['youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse']]],dtype=torch.float64)
    geometry_ids=torch.tensor([node],dtype=torch.int64)
    oldspec={**spec,'load_nodes':[h.origin_node_id for h in known.handles],
        'load_patches':[h.ordered_patch_nodes.tolist() for h in known.handles]}
    hardware=[dict(mass_kg=h.mass_kg,full_COM_inertia_kg_m2=h.com_inertia_body_kg_m2,COM_offset_m=h.com_offset_body_m) for h in known.handles]
    rows=[]
    for protocol in protocols:
        program=protocol['program_id'];directory=resolve(protocol['output'])
        execution=read(directory/'execution.json');summary=read(directory/'summary.json')
        if not (execution['terminal'] and execution['completed'] and summary['completed']
                and not Path('/proc',str(execution['pid'])).exists()):raise ValueError('Finite control incomplete/live')
        if sha(directory/'trajectory.npz')!=summary['trajectory_sha256']:raise ValueError('Finite response changed')
        if read(directory/'private-provenance/realized-hypothesis.json')!=spec or read(directory/'public-hardware.json')!=public:raise ValueError('Finite truth/hardware mismatch')
        clock=make_exciting_commands(program)
        if read(directory/'public-command-manifest.json')!=clock.public_manifest():raise ValueError('Executed known commands differ')
        finite=pure.archive(directory/'trajectory.npz');pure.exact(finite['times_s'],clock.output_times_s,'Finite clock differs')
        pure.close(finite['actual_state_times_s'],clock.output_times_s,1e-12,'Finite actual clocks differ')
        geometry,_,_,_=pure.frame_mechanics(finite,known.points,known.tets,oldspec,hardware,spec['material'])
        with torch.no_grad():
            linear=full.predict(clock,integration_hz=120)
            result=packed.predict(clock,geometry_ids,theta,device='cpu')
        modal={key:value[0] if key in ['positions_m','handle_centers_m','handle_rotation_coordinates_rad'] else value
            for key,value in result.items()}
        predictions={'full_linear':linear,'packed_cached_modal':modal}
        row={'program':program,'finite_trajectory_sha256':sha(directory/'trajectory.npz'),
            'finite_minimum_saved_J':geometry['minimum_saved_J'],'finite_patch_distance_defect_m':patch_defect(finite['positions_m'],known),'controls':{}}
        for name,value in predictions.items():
            arrays={key:a.numpy() if isinstance(a,torch.Tensor) else a for key,a in value.items() if isinstance(a,(torch.Tensor,np.ndarray))}
            q=arrays['positions_m'];metrics=comparison(q,finite['positions_m'])
            metrics['finite_rigid_patch_distance_defect_m']=patch_defect(q,known)
            metrics['handle_center_coordinate_RMSE_m']=float(np.sqrt(np.mean((arrays['handle_centers_m']-finite['handle_centers_m'])**2)))
            metrics['private_step_velocity_available']='velocities_point_chord_m_s' in arrays
            if metrics['private_step_velocity_available']:
                metrics['velocity_coordinate_RMSE_m_s']=float(np.sqrt(np.mean((arrays['velocities_point_chord_m_s']-finite['velocities_point_chord_m_s'])**2)))
            else:metrics['velocity_scope']='Frozen packed interface returns q/poses only; private120Hz chord velocities unavailable.'
            destination=out/(program+'__'+name+'.npz');np.savez_compressed(destination,**arrays)
            metrics['prediction_sha256']=sha(destination);row['controls'][name]=metrics
        row['modal_vs_full_linear']=comparison(modal['positions_m'].numpy(),linear['positions_m'].numpy())
        rows.append(row);print(json.dumps(row),flush=True)
    report={'status':'COMPLETED_UNKNOWN_POINT_SUPPLIED_TRUE_PARAMETER_FORWARD_CONTROLS','rows':rows,'elapsed_s':time.monotonic()-started,
        'frozen_marker_sha256':sha(marker_path),'reference_summaries_sha256':marker['reference_summaries_sha256'],
        'design_sha256':DESIGN_SHA,'exact_frozen_cache_sha256':CACHE_SHA,'source_sha256':sha(__file__),
        'frozen_joint_source_sha256':marker['source_sha256'],'truth_and_held_reads_after_gate_only':True,
        'predictions_per_model_per_program':1,'CPU_threads':2,'GPU_used':False,'cache_rebuilt':False,'fit_or_tuning':False,
        'finite_or_reference_reruns':0,'calibration_claim':False,
        'limitation':'Supplied true node/material compares forward approximation with the finite reference independently of inference. Packed output has no120Hz chord velocity; no20Hz difference is substituted. Finite/native/continuum equivalence not claimed.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();run(args.output)
