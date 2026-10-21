"""Reproducible fresh unknown-point-support TRAIN inputs, never responses.

Public prior/hardware/queries/commands are separate from private retained truth.
Only category rejection BC0 is permitted; no support/response-based selection.
Finite protocols use the unchanged validated worker but are NOT launched.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
try:
    import campaign_hypothesis_dynamics as worker
    from campaign_hypothesis_inputs import public_hardware_contexts,make_prior,conditioned_BC_sample,realize_hypothesis,plain
    from campaign_material_marginal import MaterialMarginalPrior
    from campaign_support_mask_prior import support_mask_distribution
    from campaign_exciting_commands import PROGRAM_IDS,make_exciting_commands
    from campaign_sparse_depth_queries import make_queries
except ImportError:
    from experiments import campaign_hypothesis_dynamics as worker
    from experiments.campaign_hypothesis_inputs import public_hardware_contexts,make_prior,conditioned_BC_sample,realize_hypothesis,plain
    from experiments.campaign_material_marginal import MaterialMarginalPrior
    from experiments.campaign_support_mask_prior import support_mask_distribution
    from experiments.campaign_exciting_commands import PROGRAM_IDS,make_exciting_commands
    from experiments.campaign_sparse_depth_queries import make_queries

ROOT=worker.ROOT;SOURCE=Path(__file__).resolve();SEED=20261110
ACQUIRED_SHA='9699169b51e02184f6eb41372d4c00dcc8261eda844f8d712b2b2452f2b677e1'
HELD_SHA='e6bc8b2c146c9de1abcb12eb4b6972fbec4ab25fbccab44b651cc6621fac6897'


def write(path,value,private=False):
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
    if private:Path(path).chmod(0o600)


def record(path):return {'path':Path(path).resolve().relative_to(ROOT).as_posix(),'sha256':worker.sha(path)}


def public_inputs(public):
    """No latent state/seed/response argument can enter public conditioning."""
    prior=make_prior(public)
    acquired=make_queries(public['rest_points_m'],public['surface_faces'],frame_indices=(10,20),offsets_deg=((0.,0.),),grid_size=6)
    held=make_queries(public['rest_points_m'],public['surface_faces'],frame_indices=(30,50),offsets_deg=((60.,10.),),grid_size=6)
    return prior,acquired,held


def build(output,finite_prefix):
    torch.set_num_threads(2)
    destination=worker.resolve(output);finite=worker.resolve(finite_prefix)
    if destination.exists() or finite.exists():raise ValueError('Fresh input and response prefixes required; never overwrite or select completed cases')
    destination.mkdir(parents=True);public_dir=destination/'public';public_dir.mkdir()
    private=destination/'private-truth';private.mkdir(mode=0o700);protocol_dir=destination/'finite-protocols';protocol_dir.mkdir()
    # Known static H0 is fixed BEFORE any material/support/category draw.
    public=public_hardware_contexts()[0];prior,acquired,held=public_inputs(public)
    public_path=public_dir/'H0.json';write(public_path,public)
    acquired_path=public_dir/'acquired-queries.json';write(acquired_path,acquired)
    held_path=public_dir/'held-queries.json';write(held_path,held)
    if worker.sha(acquired_path)!=ACQUIRED_SHA or worker.sha(held_path)!=HELD_SHA:raise ValueError('Rest-only public query recipe changed')
    distribution=support_mask_distribution(prior);distribution_path=public_dir/'point-mask-prior.json';write(distribution_path,distribution)
    known_commands={}
    for program in PROGRAM_IDS:
        clock=make_exciting_commands(program);path=public_dir/(program+'.npz')
        np.savez_compressed(path,**{k:getattr(clock,k) for k in ['endpoint_times_s','output_times_s','handle_forces_world_N',
            'handle_torques_world_Nm','nodal_forces_world_N','body_acceleration_world_m_s2']})
        known_commands[program]=record(path)
    public_manifest={'schema':'unknown-point-public-prior-query-control-v0.1','role':'public_known_inputs_no_true_material_or_support_location',
        'public_hardware':record(public_path),'conditioned_boundary_class':'point_pin','point_mask_prior':record(distribution_path),
        'material_prior':MaterialMarginalPrior(prior).metadata(),'prior_sha256':prior.metadata()['prior_sha256'],
        'public_acquired_queries':record(acquired_path),'public_held_queries':record(held_path),'known_commands':known_commands,
        'acquisition_frames':[10,20],'acquisition_times_s':[.5,1.],'acquisition_rays':72,
        'held_recipe':{'frame_indices':[30,50],'offsets_deg':[[60.,10.]],'grid_size':6},
        'public_actuator_hz':120,'seconds':3,'outputs':61,'output_hz':20,
        'B_semantics':'Known reset-force program B, not a sequential continuing poke.',
        'latent_truth_fields_present':False,'operator_provenance_and_solver_settings_are_not_neural_features':True}
    manifest_path=public_dir/'manifest.json';write(manifest_path,public_manifest)
    # One full retained-state draw conditional only on BC0, no mask selection.
    state,sampling=conditioned_BC_sample(prior,0,np.random.default_rng(SEED))
    spec=realize_hypothesis(public,state,prior)
    if spec['fixture_nodes'][0] not in {m['node_id'] for m in distribution['masks']}:raise ValueError('Truth decoder falls outside exactprior support; abort, do not redraw')
    truth_state_path=private/'retained-state.json';write(truth_state_path,{
        'role':'private_truth_for_finite_input_generation_only_not_posterior',
        'seed':SEED,'sampling':sampling,'prior_sha256':prior.metadata()['prior_sha256'],
        'state':plain(vars(state)),'material_and_location_jointly_retained':True},True)
    truth_spec_path=private/'realized-hypothesis.json';write(truth_spec_path,spec,True)
    physics_cases=[]
    for label,program in zip(['A','B'],PROGRAM_IDS):
        protocol=dict(protocol='independent-hypothesis-fixed-clock-smoke-v0.1',status='FROZEN_REVIEWED_SINGLE_TRAIN_SMOKE',
            split='TRAIN',cell_id='fresh_unknown_point_'+label,input_design=record(manifest_path),
            settings=worker.SETTINGS,information_roles=worker.INFORMATION_ROLES,integration_hz=120,
            output=finite_prefix+'/reference-'+label,public_hardware=record(public_path),realized_hypothesis=record(truth_spec_path),
            known_commands=known_commands[program],program_id=program,public_command_manifest=make_exciting_commands(program).public_manifest(),
            source_sha256=worker.runtime_sources(),resources={'cpu_workers':1,'threads_per_worker':2,'gpu_used':False},
            authorization='INPUTS_ONLY_NOT_LAUNCH_AUTHORIZATION; finite truth generation requires separate root authorization.',
            qualification='FreshTRAIN unknownpoint-location truth. Inputgeneration only; no mechanics/sensor/predictive/inference qualification.')
        path=protocol_dir/('reference-'+label+'-protocol.json');write(path,protocol);artifact=record(path)
        # Pure existing guard, never worker.run/build_model/advance/step.
        worker.validate_protocol(artifact['path'],artifact['sha256']);physics_cases.append(artifact)
    sources=worker.runtime_sources()
    for name in ['prepare_unknown_point_oracle.py','campaign_support_mask_prior.py','campaign_material_marginal.py',
                 'campaign_sparse_depth_queries.py','render_independent_trajectory.py','campaign_ray_sensor.py','censored_range_observation.py']:
        sources['experiments/'+name]=worker.sha(ROOT/'experiments'/name)
    reference_plan={'counts_per_mask':[256,1024],'seeds':[20261112,20261113,20261114],
        'stratified_weight':'exact p(mask)/Nmask for each material-prior sample; no facearea/category renormalization.',
        'mask_order':'ascending globalnode_id','material_rng':'ONE material_prior.sample(55*1024,default_rng(seed)); geometry_ids=repeat(sorted_global_node_ids,1024). First256 rows IN EACHmask are nestedrefinementprefix.',
        'mask_count':len(distribution['masks']),'posterior_predictive_particles':2048,'posterior_predictive_seed':20261115,
        'posterior_predictive_reference_seed':20261112,'count_or_seed_selection_after_likelihood':False,
        'prior_predictive_particles':2048,'prior_predictive_seed':20261118,
        'held_observation_seeds':dict(zip(PROGRAM_IDS,[20261116,20261117])),
        'held_predictive_evaluation':'Afterposteriorfreeze, compare equalbudget2048prior/posteriorresamples usingproper72-ray heldpredictive logscore underthesame sensorworkingmodel; noGaussiancoverageclaim.',
        'material_prior':MaterialMarginalPrior(prior).metadata(),
        'freeze_before_truth_read':'Reference may access finiteAframes10/20 for acquisition; truth_spec, retainedstate and heldfiniteA/B stay unopened until posterior-frozen marker.'}
    design={'protocol':'unknown-point-support-material-depth-oracle-inputs-v0.1','status':'INPUTS_FROZEN_NO_FINITE_OR_INFERENCE_LAUNCH',
        'split':'TRAIN','public_hardware':record(public_path),'public_acquired_queries':record(acquired_path),'public_held_queries':record(held_path),
        'public_manifest':record(manifest_path),'truth_spec':record(truth_spec_path),'private_retained_state':record(truth_state_path),
        'physics_cases':physics_cases,'source_sha256':sources,'material_prior':MaterialMarginalPrior(prior).metadata(),
        'support_prior':{'distribution':record(distribution_path),'prior_sha256':prior.metadata()['prior_sha256'],
            'boundary_index':0,'mask_count':55,'exact_total_probability':distribution['total_probability_exact'],
            'conditional_continuous_center_regions_retained':True,'mechanics_depends_on_node_only':True},
        'likelihood':{'range_std_m':.0002,'visibility_error_probability':.005,'background_std_m':.05,'outlier_probability':0.,
            'covariance':'Independent range noise; absent ranges censored','observation_seed':20261111},
        'reference_plan':reference_plan,'truth_sampling':{'seed':SEED,'method':'First complete fullprior state accepted by BC0categoryrejection, cap10000; no node/response selection.'},
        'truth_provenance_is_not_neural_input':True,'B_semantics':'Freshreset known programB, not sequentialpoke.',
        'reproduction_command':f'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/prepare_unknown_point_oracle.py --output {output} --finite-prefix {finite_prefix}',
        'finite_runs_launched':0,'inference_runs_launched':0,'raw_truth_publish_allowed':False}
    design_path=destination/'protocol-design.json';write(design_path,design)
    summary={'status':'INPUTS_FROZEN_NO_RUNS_LAUNCHED','design':record(design_path),'public_manifest':record(manifest_path),
        'source':record(SOURCE),'finite_cases':physics_cases,'reference_plan':reference_plan,'public_query_hashes_verified':True,
        'private_truth_generated_but_not_in_public_features':True,'selected_truth_node_or_material_published':False,
        'truth_raw_files_do_not_publish':True,'reproduction_command':design['reproduction_command']}
    write(destination/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--finite-prefix',required=True);args=p.parse_args()
    summary=build(args.output,args.finite_prefix);print(json.dumps({'status':summary['status'],'design':summary['design'],'finite_runs_launched':0,'inference_runs_launched':0}))
