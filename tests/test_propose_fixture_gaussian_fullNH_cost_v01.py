from experiments import propose_fixture_gaussian_fullNH_cost_v01 as m


def test_exact660_catalog_retains_all20_groups_11loads_3meshes_no_repeats():
    rows=m.catalog();assert len(rows)==660 and len({x['id'] for x in rows})==660
    assert len({x['group'] for x in rows})==20
    assert all(sum(x['group']==g for x in rows)==33 for g in {x['group'] for x in rows})
    assert {x['n'] for x in rows}=={3,6,12}


def test_factor_sharing_and_reuse_cost_credit_do_not_remove_live_factor_cost():
    result=dict(meshes={str(n):dict(initialization_s=10.) for n in (3,6,12)},rows=[dict(n=n,NH80_solve_s=2.,linear_warm_s=1.,independent_checks_and_capture_s=3.) for n in (3,6,12) for _ in range(2)])
    c=m.costs(result)
    assert c['new_rest_factors']==60 and c['new_fields']==654
    assert c['estimated_initialization_s']==600 and c['estimated_full_payload_s']==3960
    assert c['measured_reused_payload_credit_s']==36 and c['estimated_new_serial_s']==4524
