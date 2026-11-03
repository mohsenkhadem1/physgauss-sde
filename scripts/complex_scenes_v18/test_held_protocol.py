from scripts.complex_scenes_v18.held_protocol import draft
from scripts.complex_scenes_v18.scenarios import FAMILIES
from scripts.complex_scenes_v18.world_queries import Query


def test_draft_preserves_all_scenarios_and_separates_changed_queries():
    p=draft();assert p['status']=='draft_not_frozen'and not p['held_data_generated']
    assert {r['family']for r in p['families']}==set(FAMILIES)
    assert sum(r['worlds']for r in p['families'])==24
    assert sum(r['worlds']*len(r['queries'])for r in p['families'])==48
    for row in p['families']:
        calibration=Query(**row['calibration'])
        assert all(Query(**q)!=calibration for q in row['queries'])
        assert len({str(q)for q in row['queries']})==2
