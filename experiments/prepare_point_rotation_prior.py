"""NEW independent point-rotation cohort: inputs only, never physics.

All48 complete prior states are drawn before any response. Role labels are
operator provenance, never public neural features. Reserved truth is written
only for finite input generation and must remain closed during calibration.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
try:
    from prepare_unknown_point_oracle import (worker,ROOT,write,record,public_inputs,
        public_hardware_contexts,conditioned_BC_sample,realize_hypothesis,plain,
        MaterialMarginalPrior,support_mask_distribution,PROGRAM_IDS,make_exciting_commands,
        ACQUIRED_SHA,HELD_SHA)
except ImportError:
    from experiments.prepare_unknown_point_oracle import (worker,ROOT,write,record,public_inputs,
        public_hardware_contexts,conditioned_BC_sample,realize_hypothesis,plain,
        MaterialMarginalPrior,support_mask_distribution,PROGRAM_IDS,make_exciting_commands,
        ACQUIRED_SHA,HELD_SHA)

SOURCE=Path(__file__).resolve()
SCHEMA='independent-point-rotation-prior-cohort-v0.2'
ROLES=('CALIBRATION','RESERVED_EVALUATION')
SEED_BASE=2026120100
WORKER_SHA='595143a5a6034b3d7373c90b92a71fec111ba8f58d78753d95bc45bdeaef601d'


def identities():
    return [dict(object_id=f'point_prior_{i:03d}',index=i,seed=SEED_BASE+i,
                 role=ROLES[int(i>=16)]) for i in range(48)]


def role_cases(plan,role):
    """Only calibration data is exposed by this interface; no implicit release."""
    if role!='CALIBRATION':raise ValueError('Reserved evaluation requires a new separately frozen evaluation protocol; no access through calibration interface')
    return [c for c in plan['cells'] if c['role']==role]


def retained_payload(identity,state,sampling,prior_sha256):
    """Retain the complete state, including inactive/discrete nuisance fields."""
    raw=plain(vars(state))
    if np.asarray(raw['coordinates']).shape!=(1,12):
        raise ValueError('Complete12D single-object prior draw required')
    return dict(**identity,sampling=sampling,state=raw,prior_sha256=prior_sha256,
        role_of_truth='FINITE_INPUT_GENERATION_ONLY_UNTIL_ROLE_RELEASE',complete_state_dimension=12)


def sources():
    pins=worker.runtime_sources()
    if pins['experiments/campaign_hypothesis_dynamics.py']!=WORKER_SHA:raise ValueError('Validated595 worker required unchanged')
    for name in ['prepare_point_rotation_prior.py','run_point_rotation_prior.py',
                 'prepare_unknown_point_oracle.py','campaign_support_mask_prior.py',
                 'campaign_material_marginal.py','campaign_sparse_depth_queries.py',
                 'render_independent_trajectory.py','campaign_ray_sensor.py','censored_range_observation.py']:
        pins['experiments/'+name]=worker.sha(ROOT/'experiments'/name)
    return pins


def build(output,finite_prefix):
    torch.set_num_threads(2)
    destination=worker.resolve(output);finite=worker.resolve(finite_prefix)
    if destination.exists() or finite.exists():raise ValueError('Fresh inputs/response prefix required; no redraw/retry/overwrite')
    destination.mkdir(parents=True);public_dir=destination/'public';public_dir.mkdir()
    private=destination/'private-truth';private.mkdir(mode=0o700)
    protocols=destination/'finite-protocols';protocols.mkdir()
    for role in ROLES:(private/role).mkdir(mode=0o700)
    # Fixed public hardware/prior and common control/query clock BEFORE all draws.
    public=public_hardware_contexts()[0];prior,acquired,held=public_inputs(public)
    public_path=public_dir/'H0.json';write(public_path,public)
    ap=public_dir/'acquired-queries.json';hp=public_dir/'held-queries.json'
    write(ap,acquired);write(hp,held)
    if worker.sha(ap)!=ACQUIRED_SHA or worker.sha(hp)!=HELD_SHA:raise ValueError('Frozen rest-only72-ray recipes changed')
    distribution=support_mask_distribution(prior);dp=public_dir/'point-mask-prior.json';write(dp,distribution)
    commands={}
    for program in PROGRAM_IDS:
        clock=make_exciting_commands(program);path=public_dir/(program+'.npz')
        np.savez_compressed(path,**{k:getattr(clock,k) for k in ['endpoint_times_s','output_times_s',
            'handle_forces_world_N','handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2']})
        commands[program]=record(path)
    manifest=dict(schema='point-rotation-cohort-public-known-inputs-v0.2',public_hardware=record(public_path),
        conditioned_boundary_class='point_pin',full_prior=prior.metadata(),material_prior=MaterialMarginalPrior(prior).metadata(),
        support_prior=record(dp),public_acquired_queries=record(ap),public_held_queries=record(hp),known_commands=commands,
        public_actuator_hz=120,output_hz=20,seconds=3,frames=61,
        acquired_frames=[10,20],acquired_times_s=[.5,1.],acquired_ray_count=72,
        held_frames=[30,50],held_ray_count=72,B_semantics='Fresh reset-force program; not a sequential poke.',
        true_material_or_support_location_present=False,operator_role_seed_and_integration_metadata_are_not_neural_features=True)
    mp=public_dir/'manifest.json';write(mp,manifest)
    cells=[];objects=[]
    for identity in identities():
        # Exactly ONE independent RNG per object; reject only BC!=0, never node/material/response.
        state,sampling=conditioned_BC_sample(prior,0,np.random.default_rng(identity['seed']))
        spec=realize_hypothesis(public,state,prior)
        if spec['fixture_nodes'][0] not in {m['node_id'] for m in distribution['masks']}:raise ValueError('Prior decoder outside support: preserve failure, no redraw')
        directory=private/identity['role']/identity['object_id'];directory.mkdir(mode=0o700)
        sp=directory/'realized-hypothesis.json';rp=directory/'retained-state.json'
        write(sp,spec,True);write(rp,retained_payload(identity,state,sampling,prior.metadata()['prior_sha256']),True)
        objects.append(dict(**identity,realized_hypothesis=record(sp),retained_state=record(rp)))
        for label,program in zip(('A','B'),PROGRAM_IDS):
            cell_id=identity['object_id']+'__'+label
            protocol=dict(protocol='independent-hypothesis-fixed-clock-smoke-v0.1',status='FROZEN_REVIEWED_SINGLE_TRAIN_SMOKE',
                split='TRAIN',cell_id=cell_id,evaluation_role=identity['role'],
                input_design=record(mp),settings=worker.SETTINGS,information_roles=worker.INFORMATION_ROLES,
                integration_hz=120,output=finite_prefix+'/cells/'+cell_id,public_hardware=record(public_path),
                realized_hypothesis=record(sp),known_commands=commands[program],program_id=program,
                public_command_manifest=make_exciting_commands(program).public_manifest(),source_sha256=worker.runtime_sources(),
                resources=dict(cpu_workers=1,threads_per_worker=2,gpu_used=False),
                authorization='INPUTS_ONLY_ROOT_MUST_AUTHORIZE_CONTROLLER_LAUNCH',
                qualification='Independent complete conditional-prior TRAIN draw. Reserved role is unavailable for calibration or diagnosis. No population calibration claim from capture alone.')
            path=protocols/(cell_id+'.json');write(path,protocol);rec=record(path)
            worker.validate_protocol(rec['path'],rec['sha256']) # hashes/metadata only, no truth parsing or mechanics
            cells.append(dict(id=cell_id,object_id=identity['object_id'],role=identity['role'],program_id=program,
                protocol=rec,output=protocol['output']))
    plan=dict(schema=SCHEMA,status='ALL_INPUTS_FROZEN_NO_FINITE_LAUNCH',cohort_version='point-rotation-prior-v02',public_manifest=record(mp),
        public_hardware=record(public_path),public_acquired_queries=record(ap),public_held_queries=record(hp),
        objects=objects,cells=cells,source_sha256=sources(),output_prefix=finite_prefix,
        sampling=dict(object_count=48,seeds=[SEED_BASE+i for i in range(48)],
            law='48 independent full ConditionalPhysicalPrior draws conditioned solely on BC0 by rejection; one RNG per seed, no stratification or reused material/support states.',
            inactive_nuisances_retained=True,coordinates_per_state=12,response_based_selection=False,
            failure_policy='Preserve failed object/cell and all queued slots; no redraw, retry, deletion or completed-case selection.'),
        roles=dict(CALIBRATION=list(range(16)),RESERVED_EVALUATION=list(range(16,48))),
        role_barrier=dict(private_truth_separate_directories=True,
            calibration_may_access_only_first16=True,reserved_truth_or_responses_for_calibration_or_diagnosis=False,
            evaluation_requires_new_frozen_protocol=True,
            controller_may_hash_all_inputs_and_terminal_artifacts_but_does_not_parse_truth_or_response_arrays=True),
        resources=dict(maximum_workers=8,threads_per_worker=2,gpu_used=False,memory_reserve_GiB=8,memory_per_worker_GiB=1),
        likelihood_or_discrepancy_fitted=False,calibration_analysis_protocol_pending=True,
        development_policy='Old48 are development. Model correction/discrepancy may use oldCALIB16 only; this new independent48 stay blind until new method/execution freezes.',
        local_object_IDs_are_namespaced_by_cohort_manifest=True,
        finite_runs_launched=0,raw_truth_responses_publish_allowed=False,
        limitation='Population calibration requires prospective inference/analysis and independent reserved evaluation. FixedH0/knownBC0/knownA-B controls only; no active-policy, native, continuum or broadBC claim.')
    path=destination/'cohort-plan.json';write(path,plan)
    summary=dict(status=plan['status'],plan=record(path),cohort_version=plan['cohort_version'],objects=48,finite_cases=96,
        seed_base=SEED_BASE,seed_count=48,complete_prior_state_dimension=12,new_physics_launched=False,
        CALIBRATION_objects=16,RESERVED_EVALUATION_objects=32,independent_prior_sampling=True,
        source_sha256=plan['source_sha256'],no_truth_values_published=True,finite_runs_launched=0,
        reproduction_command=f'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/prepare_point_rotation_prior.py --output {output} --finite-prefix {finite_prefix}')
    write(destination/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--finite-prefix',required=True)
    args=parser.parse_args();result=build(args.output,args.finite_prefix);print(json.dumps(result))
