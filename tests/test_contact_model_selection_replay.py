from scripts.contact_benchmark_v12.model_selection_replay import planned_cases


def test_replay_population_is_identity_prespecified_and_unique():
    cases=planned_cases()
    assert len(cases)==12 and len({c['name'] for c in cases})==12
    assert len([c for c in cases if 'case' in c['spec']])==8
    soft=[c for c in cases if 'case' not in c['spec']]
    assert {c['name'] for c in soft}=={f'{suite}-task{task}-demo0' for suite in ['object-soft','spatial-soft'] for task in [1,2]}
    assert all(c['cohort']=='regression' for c in soft)
