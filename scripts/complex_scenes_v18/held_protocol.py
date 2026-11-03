"""Explicit benchmark design, drafted before any held-world generation.

This module does not draw private worlds or generate test trajectories. Freezing
requires source/environment inventories and the completed development audit.
"""
from scripts.complex_scenes_v18.scenarios import FAMILIES
from scripts.complex_scenes_v18.world_queries import Query
from dataclasses import asdict


def draft():
    families=[]
    cuts=dict(clamp_release=.08,curved_push=.10,container_drop=.16,rigid_impact=.06,deformable_pair=.11,heterogeneous=.08)
    for family in FAMILIES:
        calibration=Query(family,0)
        if family=='container_drop':
            queries=[Query(family,1,initial_height_m=.015,initial_rotation_deg=12),Query(family,0,initial_height_m=.03,initial_rotation_deg=-15)]
        elif family=='deformable_pair':
            queries=[Query(family,1),Query(family,0,velocity_scale=1.2)]
        elif family=='rigid_impact':
            queries=[Query(family,1,velocity_scale=1.2,fixture_shift=.1),Query(family,0,velocity_scale=.8,fixture_shift=-.1)]
        else:
            queries=[Query(family,1,actuation_scale=.85,fixture_shift=.1),Query(family,0,actuation_scale=1.15,fixture_shift=-.1)]
        families.append(dict(family=family,worlds=4,calibration=asdict(calibration),calibration_duration_s=cuts[family],queries=[asdict(q)for q in queries]))
    return dict(status='draft_not_frozen',held_data_generated=False,
        scope='Unknown material, known boundary geometry and parameters; position-driven interactions and known gravity. Not unknown-boundary recovery or end-to-end reconstruction.',
        independent_worlds=24,queries_per_world=2,total_queries=48,families=families,
        material_distribution=dict(young_Pa_log_uniform=[8000,18000],poisson_uniform=[.25,.4],heterogeneous_contrast_log_uniform=[2,3],heterogeneous_transition_uniform=[.35,.65],two_object_materials='Independent draws; fixed across calibration and both queries',density_kg_m3=625,constitutive_family='Log-J Neo-Hookean; analytic control knows family, not parameters'),
        observations=dict(frame_dt_s=.002,visibility='Fixed material surface subset: y=0 or z=1 faces; no camera-occlusion claim',track_noise_m=0,known=['rest geometry','density','public actuator paths','gravity','collision geometry','attachment compliance'],unknown=['material parameters and spatial material field'],fit_sealed_before_query_rollouts=True),
        geometry=dict(dimensions_m=[.12,.06,.06],scope='Interaction and material transfer at fixed object geometry; no unseen-shape claim'),
        rendering=dict(scope='Object-only controlled shared appearance; environment occlusion and independent reconstruction quality not claimed',appearance_checkpoint='runs/compositional-dynamics-v17/matched-appearance-native-resolution/model/point_cloud/iteration_10000/point_cloud.ply',transport='Same author K16 transport on a common 5x5x5 material grid for every method and reference',opacity_threshold=.1,background=245/255,width=512,height=384,focal=550.,camera_target=[.10,0,-.01],camera_eyes=[[.25,-.42,.25],[-.22,-.22,.15],[.22,.30,.20]],frames_per_query=7,lpips='AlexNet v0.1; RGB converted to [-1,1]; full-image score, with foreground-union PSNR/SSIM reported separately'),
        common_physical_metrics=dict(volume='Signed tetrahedron ratios on common 7x7x7 material grid; sampled folds, not continuous validity proof',impulse='Net nongravitational body impulse from known masses and saved velocities; no individual force decomposition',contact='Same 386-point material surface; normalized implicit bowl gap and sampled vertex-to-triangle pair gap; event resolution 2 ms'),
        methods={
            'ours':dict(families=FAMILIES,calibration='Powell, 64 forward evaluations, four shared material-field parameters plus second-object scale when needed',material_network='Frozen prior decoder plus independent compressibility extension',mesh_n=1,dt_s=50e-6),
            'analytic-control':dict(families=FAMILIES,calibration='Same 64-evaluation spatial parameter fit',advantage='Oracle constitutive family',mesh_n=1,dt_s=50e-6),
            'spring-gaus':dict(families=FAMILIES,calibration='40 Adam iterations, per-edge stiffness and damping',dt_s=25e-6,adaptation='Author internal force/integrator; common boundary adapter; exact dense KNN compatibility'),
            'phystwin':dict(families=FAMILIES,calibration='40 Adam iterations, per-edge stiffness',dt_s=25e-6,adaptation='Author Warp spring force and sealed-forecast integrator; common boundary adapter; fixed zero damping'),
            'uniphy':dict(families=['container_drop'],calibration='8 fixed-seed latent-centroid initializations, then 8 AdamW iterations',dt_s=50e-6,adaptation='Released demo pretrained networks and MPM; fixed bowl/sphere collision hooks; optional adjoint bridge',contact_law_limit='Native separating-velocity projection differs from the conservative penalty boundary used by the reference and other adapters; report as a capability comparison, not matched contact-law material inference',gradient_limit='Short derivative tests remain coordinate-sensitive; disclose adaptation and incomplete qualification')},
        reference=dict(mesh_n=4,dt_s=25e-6,quadrature='lobatto4',contact_sampling='Mesh-aligned surface sampling for deformable pairs',dense_J_samples_per_axis=9,minimum_nodal_quadrature_J_guard=.01,refinement='n6 repeats for world 0 in every family and both queries; disclose mean/peak differences',qualification='Positive sampled orientation is necessary, not a convergence or nonintersection proof'),
        metrics=['full-query vector RMS','per-object rigid-aligned shape RMS','completion','sampled volume orientation','contact penetration and timing where representation supports it','body impulse where state/mass exports support it','shared-renderer PSNR/SSIM/LPIPS on fixed withheld cameras'],
        aggregation='Average two queries within world; report per-family and world-level paired differences with world bootstrap intervals; frames are not independent samples',
        failures='Retain every planned slot. Separate unsupported, fit failure, rollout failure, invalid reference and numerical-reference uncertainty. Unsupported or failed-reference slots never count as method wins.',
        freeze_requirements=['Completed development-case audit or explicit recorded terminal failure for every method-supported slot','Chosen method settings and numeric reference policy fixed','Source commit and environment/checkpoint hashes saved','Metrics and rendering protocol implemented and checked','No held reference or prediction used for tuning'])


if __name__=='__main__':
    from scripts.complex_scenes_v18.run_development import ROOT,write
    write(ROOT/'reports/complex-scenes-v18/held-protocol-draft.json',draft())
