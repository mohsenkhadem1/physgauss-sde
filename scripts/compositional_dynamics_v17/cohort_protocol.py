"""Freeze the cohort only after the final development calibration recipe selection.
No private random draws, simulated responses or forecast scoring occur here.
"""
import json
from pathlib import Path
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_dynamics_v17 import native_phystwin_cohort_adapter as native
PROTOCOL=REPORT/'fresh-cohort-protocol.json'
FREEZE=REPORT/'fresh-cohort-method-freeze.json'


def prepare():
    if PROTOCOL.exists() or FREEZE.exists():raise ValueError('Preserve existing cohort freeze')
    selection_path=REPORT/'native-budget-calibration-comparison.json'
    selection=json.loads(selection_path.read_text())
    for path,digest in selection['evidence_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Calibration selection changed')
    proposal_path=REPORT/'fresh-cohort-proposal.json';proposal=json.loads(proposal_path.read_text())
    for group in ('source_sha256','input_sha256'):
        for path,digest in proposal[group].items():
            if sha(ROOT/path)!=digest:raise ValueError('Generator proposal changed')
    # Adapter must state the selected recipe explicitly before final freeze.
    if getattr(native,'SELECTED_VARIANT',None)!=selection['selected_recipe']:raise ValueError('Native adapter has not frozen selected calibration recipe')
    shared=dict(catalogue_plan='reports/compositional-dynamics-v17/matched-material-fit-plan.json',catalogue_results='reports/compositional-dynamics-v17/matched-material-fit-results.json',
                public_root='runs/compositional-dynamics-v17/fresh-cohort/public',fit_root='runs/compositional-dynamics-v17/fresh-cohort/shared-fit',forecast_root='runs/compositional-dynamics-v17/fresh-cohort/shared-forecast',
                coarse_maxfev=24,fine_maxfev=32,coarse_n=2,fine_n=3,dt_s=50e-6,workers=8,seconds=10800,forecast_seconds=600)
    setups=[dict(name=name,setup=dict(duration_s=.30,frame_dt_s=.01,floor_z_m=z,initial_velocity_m_s=[0,0,0],initial_geometry_same=True,gravity_vector_m_s2=[0,0,-9.8]))for name,z in [('forecast-40mm',-.07),('forecast-80mm',-.11)]]
    worlds=[dict(name=name,public_directory=f"{shared['public_root']}/{name}",output_directory=f'runs/compositional-dynamics-v17/fresh-cohort/native/{name}',forecast_setups=setups)for name in proposal['generator_settings']['world_ids']]
    metrics=dict(primary_physics=['rest-area weighted world-position vector RMS','proper-rigid-aligned shape vector RMS'],secondary_physics=['reference and predicted nonrigid RMS','surface-centroid trajectory error','sampled floor penetration','closed surface volume ratio','sampled contact/rebound timing and height'],
                 render=['foreground-union PSNR','foreground-union SSIM','silhouette IoU','joint-visible depth RMSE and coverage','global PSNR/SSIM secondary'],
                 initial_appearance='Same frozen initial Gaussian checkpoint and same retained-opacity mask for both methods; native K16 versus material/F transport. Motion and transport jointly affect images. Common procedural renderer separately isolates geometry.',
                 time_sampling='31 simultaneous frames at100Hz includingt0; no time/rigid alignment in world or image scores.',
                 views='Three observed-camera poses and two held camera poses, fixed across methods. Appearance trained only on initialstate of development geometry; no future appearance optimization.',
                 aggregation='Report all four worlds and eight forecasts. Equal-weight world summary if all valid complete; otherwise primary completion coverage plus paired per-world results and clearly conditional complete-case summaries. No replacement or survivor-only primary advantage claim. Four worlds too few for strong population inference.',
                 success_interpretation='Seek lower world and shape forecast errors with comparable native-appearance quality. Report signed paired differences, no assumed win. A2dB foregroundPSNR/.02SSIM difference is a descriptive practical band, not a statistical equivalence claim.',
                 reference_scope='Finite-resolution analytic MR/Yeoh teachers; n4/n6 and timestep sensitivity on declared subset. Failed or unqualified references retained and explicitly separated from qualified evidence. Not real-world accuracy proof.')
    starts=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/cohort_generator.py',ROOT/'scripts/compositional_dynamics_v17/cohort_observations.py',ROOT/'scripts/compositional_dynamics_v17/cohort_shared_fit.py',ROOT/'scripts/compositional_dynamics_v17/cohort_pipeline.py',ROOT/'scripts/compositional_dynamics_v17/cohort_geometry_evaluation.py',ROOT/'scripts/compositional_dynamics_v17/matched_material_fit.py',ROOT/'scripts/compositional_dynamics_v17/forecast_metrics.py',ROOT/'scripts/compositional_dynamics_v17/surface_physics_metrics.py',ROOT/'scripts/compositional_dynamics_v17/render_metrics.py',ROOT/'scripts/compositional_dynamics_v17/gaussian_transport.py']
    sources={str(p.relative_to(ROOT)):sha(p)for p in closure(starts)}
    sources.update(proposal['source_sha256']);sources.update(native.source_inventory())
    for name in ['scripts/compositional_dynamics_v17/test_cohort_shared_fit.py','scripts/compositional_dynamics_v17/test_cohort_pipeline.py','tests/test_surface_physics_metrics_v17.py','tests/test_cohort_geometry_gate_independent_v17.py']:
        sources[name]=sha(ROOT/name)
    inputs=dict(proposal['input_sha256']);inputs[str(selection_path.relative_to(ROOT))]=sha(selection_path)
    for name in ('catalogue_plan','catalogue_results'):inputs[shared[name]]=sha(ROOT/shared[name])
    appearance_status=ROOT/'runs/compositional-dynamics-v17/matched-appearance-native-resolution/training-complete.json'
    appearance=json.loads(appearance_status.read_text());checkpoint=ROOT/appearance['checkpoint']
    if not appearance['complete'] or sha(checkpoint)!=appearance['checkpoint_sha256']:raise ValueError('Initial appearance must already be committed')
    for path in [appearance_status,checkpoint,REPORT/'matched-appearance-native-resolution-design.json',REPORT/'native-final-recipe.json']:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    metrics['mass_discretization_scope']='Both methods receive known0.27kg and uniform canonical geometry. Native125equal-mass q5points have1.5times continuum rest rotational inertia; Q2 uses consistent uniform-density mass. Native729capacity control has1.25times continuum inertia. This is a configuration-level comparison, not causal isolation of constitutive law; elastic modes/contact response can reflect mass discretization. No rigidrotation challenge claimed.'
    metrics['appearance_checkpoint']=appearance['checkpoint']
    metrics['appearance_checkpoint_sha256']=appearance['checkpoint_sha256']
    metrics['camera_design']='reports/compositional-dynamics-v17/matched-appearance-native-resolution-design.json'
    # Catalogue plan transitively seals shared checkpoints and original forward inputs.
    catalogue=json.loads((ROOT/shared['catalogue_plan']).read_text())
    for group in ('source_sha256','input_sha256'):
        for path,digest in catalogue[group].items():
            if sha(ROOT/path)!=digest:raise ValueError('Catalogue dependency changed')
            (sources if group=='source_sha256' else inputs)[path]=digest
    protocol=dict(proposal,status='frozen',method_recipe=dict(shared=shared,native_variant=selection['selected_recipe']),native_recipe=native.RECIPE,native_source_sha256=native.source_inventory(),native_worlds=worlds,metrics=metrics,source_sha256=sources,input_sha256=inputs)
    write(PROTOCOL,protocol)
    write(FREEZE,dict(protocol_sha256=sha(PROTOCOL),source_sha256=sources,input_sha256=inputs,scope='Frozen before fresh private draws; public-only inference and committed predictions precede target scoring. Raw datasets remain local.'))
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),method_freeze_sha256=sha(FREEZE))))

if __name__=='__main__':prepare()
