"""Prospective static prediction/probing study, before any new outcomes.

Catalog admission is separate from the two central-force numerical repair.
Every action and target needs physics qualification before actual inference.
The matching-linear and nonlinear-data diagnostics must remain separate.
"""
import numpy as np
from experiments.campaign_surface_force_study_v01 import observation_points,SIGMA_M

ACTION_CENTERS=(.09,.18,.27)
TARGET_CENTERS=(.135,.225)
CATALOG_CENTERS=(.09,.135,.18,.225,.27)
COHORT_SEED=2026102481
PREFIX_NOISE_SEED=2026102482
PROBE_NOISE_SEED=2026102483
RANDOM_ACTION_SEED=2026102484


def load_catalog(n):
    from experiments.campaign_surface_p2_static_v01 import surface_traction
    loads={}
    for center in CATALOG_CENTERS:
        label=f'x{round(center*1000):03d}'
        loads['normal_'+label]=surface_traction(n,[0,0,-1],.10,center_x=center)[0]
        loads['paired_axial_'+label]=sum(surface_traction(n,[1,0,0],.24,center_x=center,face=face)[0] for face in ('top','bottom'))
    return loads


def design():
    targets=[family+f'_x{round(center*1000):03d}' for family in ('normal','paired_axial') for center in TARGET_CENTERS]
    return dict(schema='surface-p2-study-design-v01',status='PROSPECTIVE_NO_OUTCOMES',
        prior='Independent uniform logE[log2000,log8000], logK[log5,log30]; known Gaussian center/width and both-end clamps',
        unknowns=['Young_modulus','compliant_support_strength'],prefix='normal_x180',target_loads=targets,
        normal_actions=['normal_'+f'x{round(c*1000):03d}'for c in ACTION_CENTERS],
        axial_actions=['paired_axial_'+f'x{round(c*1000):03d}'for c in ACTION_CENTERS],
        actions_and_targets_disjoint=True,catalog_load_count=10,
        physics='All10physicaltractions×same5priorcaseprotocol×all3P2meshes must qualify; central2force study cannot admit catalog automatically',
        sensors_m=observation_points().tolist(),channels=24,noise_SD_m=SIGMA_M,
        objects=16,cohort_seed=COHORT_SEED,prefix_noise_seed=PREFIX_NOISE_SEED,
        probe_noise_seed=PROBE_NOISE_SEED,random_action_seed=RANDOM_ACTION_SEED,
        object_generation='Independent original-prior unit draws; keep all16, no truth filtering or redraws; arraysLOCAL',
        prefix_replicates=1,post_probe_noise_replicates=3,
        paired_noise='Same prefix and same3independent24-channel postnoise vectors for every method/action withinobject; clusters are16objects, not48replicates',
        data_laws=['matching_linear_finite_model','nonlinear_80point_finite_model_stress'],
        data_laws_scope='Separate results. Matching-linear validates only conditional numerical inference. NH-data stress assesses model discrepancy. Neither is real physical calibration.',
        inference='Full-mode exact linear finite model; Simpson129/257/513/1025 and independent positiveGauss128/256/512; stop unresolved rather than chooseconvenientlevel',
        quadrature_gates=dict(log_evidence_difference=.01,maximum_CDF_difference=.005,maximum_unit_quantile_difference=.002,prediction_absolute_tolerance_m=2e-6,prediction_relative_SD=.01,Gauss_partial_CDF_at_Simpson_05_50_95_tolerance=.005),
        posterior_diagnostics=['evidence','intrinsicCDF','intrinsicquantiles','targetmean','targetSD','independentGauss_partialCDF','boundarymass'],
        prediction_methods=['true_parameter_linear_oracle','intrinsic_MAP_plugin','posterior_predictive_mean','prior_predictive_mean'],
        oracle='Known trueE/K and exactsameknownforce; nonlineartruthversuslinearoracle explicitly measuresforwardmodelerror',
        policy='Covariance best-linear-predictor targetMSE reduction, sealed from prefix posterior before actualpostmeasurements',
        policy_baselines=['each_of_three_prospective_fixed_actions','uniform_random_exact_average_over_actions','Gaussian_observation_information'],
        force_budget='Compare normalonly and pairedaxialonly separately; sameforce/area withinfamily, not equalwork; no mixedfamilycost claim',
        score='Unprobed/disjoint target latent24channel responses across4loads; selectedaction displacement never joins heldoutscore',
        uncertainty='90%intrinsicE/K intervals and predictiveSD; report narrowing withtruthcoverage/error, notconfidencealone',
        model_discrepancy='Report whitenedNH-minus-linearoracle error perobserved24channel and fulltarget;20umbulk gate isnotlikelihoodfidelity',
        comparison='Pairedobjectmeanerrorchanges andobjectbootstrap95%CI; compareagainstALL3fixedpolicies beforeclaimingfixed-policyadvantage; random andinformation baselines separated',
        interpretation='Static idealtracked3Dpoint prediction; noRGBDperception, missingviewreconstruction, dynamics, contact, drop, realphysicalcalibration orfuturemotionclaim',
        no_automatic_execution=True)
