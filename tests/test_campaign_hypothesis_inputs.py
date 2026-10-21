"""Pure prior/mesh/manifest checks; never construct or step a dynamics model."""
from copy import deepcopy
from collections import Counter
import math
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import campaign_hypothesis_inputs as inputs

PROGRAMS=['exciting_world_a_v01','exciting_world_b_v01']


@pytest.fixture(scope='module')
def manifest():return inputs.make_manifest(program_ids=PROGRAMS,seed=20261022)


def test_full_fixed_design_and_predeclared_fit_development_roles(manifest):
    assert manifest['hypothesis_count']==len(manifest['hypotheses'])==48
    assert manifest['capture_count']==len(manifest['cells'])==96
    assert len({c['id'] for c in manifest['cells']})==96
    assert Counter(c['evaluation_role'] for c in manifest['cells'])=={'fit':32,'development_force_transfer':32,'unseen_middle_material':32}
    assert all(c['split']=='train' and c['status']=='DECLARED_NOT_EXECUTED' for c in manifest['cells'])
    assert all({c['program_id'] for c in manifest['cells'] if c['hypothesis_id']==h['id']}==set(PROGRAMS) for h in manifest['hypotheses'])
    assert manifest['SBC_claim'] is manifest['production_dataset_admitted'] is False


def test_public_H_first_has_full_hardware_and_no_source_material_or_BC_truth(manifest):
    for public in manifest['public_hardware']:
        assert 'material' not in public and 'boundary_kind' not in public and 'fixture_nodes' not in public
        assert 'recipe' not in public and 'boundaries' not in public
        known=inputs.known_hardware_from_public(public)
        assert len(known.handles)==2 and all(len(h.ordered_patch_nodes)==9 for h in known.handles)
        assert all(h.com_inertia_body_kg_m2.shape==(3,3) for h in known.handles)
        assert inputs.physical_hardware_digest(known)==public['physical_hardware_sha256']
        assert len([h for h in manifest['hypotheses'] if h['public_hardware_id']==public['id']])==12


def test_material_log_strata_shared_globally_and_other_original_marginals(manifest):
    bounds=manifest['material_stratification']['log_E_bounds'];material=manifest['shared_material_intrinsic_coordinates']
    for mi,x in enumerate(material):
        assert bounds[mi]<=x[0]<bounds[mi+1]
        assert 400<=math.exp(x[1])<=850 and .28<=x[2]<=.42 and .35<=math.exp(x[3])<=.85
        for h in manifest['hypotheses']:
            if h['material_tuple_id']=='m'+str(mi):
                np.testing.assert_array_equal(h['spec']['retained_prior_state']['coordinates'][0][:4],x)
                assert h['E_stratum_conditional_density_factor']==3.
                assert h['individual_conditional_BC_and_E_stratum_log_density']==pytest.approx(h['individual_base_prior_log_density']+math.log(12),abs=1e-14)
    assert manifest['grouping'].find('not iid')>=0


def test_support_motion_and_all_nuisances_fixed_across_material_counterfactuals(manifest):
    for hid in ['H0','H1','H2','H3']:
        for bc in range(4):
            group=[h for h in manifest['hypotheses'] if h['public_hardware_id']==hid and h['id'].endswith('__bc'+str(bc))]
            assert len(group)==3
            first=group[0]['spec']
            for h in group[1:]:
                s=h['spec']
                for key in ['fixture_nodes','support_node_groups','support_node_axis_masks','support_origins_world_m','motion_amplitude_m','motion_direction_world','prior_sha256']:
                    assert s[key]==first[key]
                for key in ['boundary_type','force_type','support_face','support_cells','load_faces']:
                    assert s['retained_prior_state'][key]==first['retained_prior_state'][key]
                np.testing.assert_array_equal(np.array(s['retained_prior_state']['coordinates'])[:,4:],np.array(first['retained_prior_state']['coordinates'])[:,4:])


def test_actual_support_counts_clearance_and_independent_reconstruction(manifest):
    publics={h['id']:h for h in manifest['public_hardware']}
    for h in manifest['hypotheses']:
        public=publics[h['public_hardware_id']];known=inputs.known_hardware_from_public(public);s=h['spec'];bc=int(s['retained_prior_state']['boundary_type'][0])
        assert inputs.validate_realized_hypothesis(known,s)==s
        assert [len(g) for g in s['support_node_groups']]==[[1,0],[16,0],[9,9],[16,0]][bc]
        mask=np.array(s['support_node_axis_masks']);assert mask.shape==(2,len(known.points),3)
        assert np.array_equal(mask[:,:,0],mask[:,:,1]) and np.array_equal(mask[:,:,1],mask[:,:,2])
        assert np.array_equal(np.flatnonzero(mask.any((0,2))),s['fixture_nodes'])
        hardware=np.unique(np.concatenate([a.ordered_patch_nodes for a in known.handles]))
        assert np.min(np.linalg.norm(known.points[s['fixture_nodes'],None]-known.points[hardware][None],axis=-1))>.027
        assert len(s['retained_prior_state']['coordinates'][0])==12
        assert s['capacities']=={'S':2,'H':2} and s['known_programs_override_passive_force_nuisances']


@pytest.mark.parametrize('field',['material','fixture_nodes','boundary_kind','support_law','physical_hardware_sha256'])
def test_changed_physical_hypothesis_cannot_bypass_intrinsic_realization(manifest,field):
    s=deepcopy(manifest['hypotheses'][0]['spec']);known=inputs.known_hardware_from_public(manifest['public_hardware'][0])
    if field=='material':s[field]['density_kg_m3']*=1.01
    elif field=='fixture_nodes':s[field]=[0]
    elif field=='boundary_kind':s[field]='moving_patch'
    elif field=='support_law':s[field]['kind']='invented_interpolation'
    else:s[field]='0'*64
    with pytest.raises(ValueError):inputs.validate_realized_hypothesis(known,s)


def test_complete_mass_inertia_H_mismatch_rejected(manifest):
    h=deepcopy(manifest['public_hardware'][0]);h['handles'][0]['mass_kg']*=2
    known=inputs.known_hardware_from_public(h)
    with pytest.raises(ValueError,match='hardware'):inputs.validate_realized_hypothesis(known,manifest['hypotheses'][0]['spec'])


def test_analytic_support_law_rest_velocity_and_time_derivative(manifest):
    moving=next(h['spec'] for h in manifest['hypotheses'] if h['spec']['boundary_kind']=='moving_patch')
    np.testing.assert_array_equal(inputs.support_translation(moving,0),np.zeros(3))
    np.testing.assert_array_equal(inputs.support_velocity(moving,0),np.zeros(3));np.testing.assert_array_equal(inputs.support_velocity(moving,3),np.zeros(3))
    assert .006<=moving['motion_amplitude_m']<=.010
    np.testing.assert_allclose(np.linalg.norm(moving['motion_direction_world']),1.,rtol=0,atol=2e-15)
    t=.87;eps=1e-6
    np.testing.assert_allclose((inputs.support_translation(moving,t+eps)-inputs.support_translation(moving,t-eps))/(2*eps),inputs.support_velocity(moving,t),rtol=2e-10,atol=1e-12)
    for h in manifest['hypotheses']:
        if h['spec']['boundary_kind']!='moving_patch':np.testing.assert_array_equal(inputs.support_translation(h['spec'],1.2),np.zeros(3))


def test_conditioned_category_budget_aborts_not_redraws_or_drops():
    class Prior:
        calls=0
        def sample(self,n,rng):
            self.calls+=1;return SimpleNamespace(boundary_type=np.array([1]))
    p=Prior()
    with pytest.raises(RuntimeError,match='abort'):inputs.conditioned_BC_sample(p,0,np.random.default_rng(1),maximum_draws=3)
    assert p.calls==3


def test_manifest_never_calls_old_25D_decode_and_is_reproducible(manifest,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Old25D decode forbidden')
    monkeypatch.setattr(inputs.ConditionalPhysicalPrior,'decode',forbidden)
    monkeypatch.setattr(inputs.ConditionalPhysicalPrior,'decode_torch',forbidden)
    repeat=inputs.make_manifest(program_ids=PROGRAMS,seed=20261022)
    assert repeat==manifest


def test_larger_stratified_count_is_bounded_and_no_unapproved_fit_split():
    four=inputs.make_manifest(program_ids=PROGRAMS,seed=20261022,material_draw_count=4)
    assert four['hypothesis_count']==64 and four['capture_count']==128
    assert four['material_stratification']['conditional_density_factor']==4.
    assert len(four['material_stratification']['log_E_bounds'])==5
    assert four['fit_and_development_counts']=={'future_design_unassigned':128}
    for bad in [0,33,True]:
        with pytest.raises(ValueError,match='Bounded'):inputs.make_manifest(program_ids=PROGRAMS,material_draw_count=bad)
