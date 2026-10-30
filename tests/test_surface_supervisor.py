from scripts.coupled_contact_v13 import surface_run_campaign,surface_finish,surface_render,render
from scripts.coupled_contact_v13 import engineered_scenarios,scenarios


def test_surface_stages_exact_and_bounded_scope():
    stages=surface_run_campaign.stages()
    assert [n for n,_ in stages]==['library','development-reference','development-evaluate','reserved-reference','reserved-evaluate']
    assert [n for n,_ in surface_finish.STAGES]==['surface_ablation','surface_report','surface_verify','verify_surface_ablation','surface_render']
    assert surface_run_campaign.BASE==surface_finish.BASE
    assert surface_run_campaign.remaining is surface_finish.remaining


def test_surface_renderer_isolation_and_fixed_cases():
    original=(render.BASE,render.REPORT,render.build,render.command)
    module=surface_render.implementation()
    assert module.BASE==surface_render.BASE and module.REPORT==surface_render.REPORT
    assert module.build is engineered_scenarios.build and module.command is engineered_scenarios.command
    assert (render.BASE,render.REPORT,render.build,render.command)==original
    assert render.build is scenarios.build
    assert surface_render.FIXED_CASES==[['surface_test0','curved'],['surface_test3','moving_sphere'],['surface_test7','edge_drop']]


def test_workers_only_change_execution_concurrency_and_threads():
    import os
    import pytest
    import argparse
    for module in (surface_run_campaign,surface_finish):
        assert module.worker_count('12')==12
        assert module.worker_count('1')==1
        with pytest.raises(argparse.ArgumentTypeError):module.worker_count('0')
        with pytest.raises(argparse.ArgumentTypeError):module.worker_count('37')
    assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    # Worker options are execution controls, not added calibration/forecast data.
    assert surface_run_campaign.stages()==[
        ('library',['library']),('development-reference',['reference','--split','development']),
        ('development-evaluate',['evaluate','--split','development']),('reserved-reference',['reference','--split','reserved']),
        ('reserved-evaluate',['evaluate','--split','reserved'])]
