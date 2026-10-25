from experiments.audit_active_fixture_gaussian_refinement_v03 import classification


def test_physics_excitation_and_spatial_negatives_stay_separate_no_majority_rescue():
    rows=[dict(physical_passed=True,load_eligible=True) for _ in range(6)]
    spatial={name:dict(spatial_development_passed=True) for name in ('prefix','probe')}
    assert classification(rows,spatial)['qualified_only_for_six_controls']
    r=classification(rows[:5]+[dict(physical_passed=False,load_eligible=False)],spatial)
    assert not r['physical_passed'] and not r['qualified_only_for_six_controls']
    r=classification(rows,dict(spatial,probe=dict(spatial_development_passed=False)))
    assert r['physical_passed'] and not r['spatial_development_passed'] and not r['qualified_only_for_six_controls']
    r=classification(rows[:5]+[dict(physical_passed=True,load_eligible=False)],spatial)
    assert r['physical_passed'] and not r['all_loads_eligible'] and not r['qualified_only_for_six_controls']
