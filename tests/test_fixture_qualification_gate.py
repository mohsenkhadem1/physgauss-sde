import numpy as np
from scripts.phystwin.summarize_fixture import compare


def test_gate_rejects_strain_traction_and_quadrature_error_despite_small_displacement():
    p=dict(displacement_rms_tolerance_m=2e-5,displacement_max_tolerance_m=1e-4,
           strain_relative_L2_tolerance=.05,traction_relative_L2_tolerance=.02,
           energy_relative_tolerance=.01,force_balance_tolerance_N=1e-8,
           moment_balance_tolerance_Nm=1e-9,nonlinear_residual_tolerance_N=1e-9,
           quadrature_displacement_rms_tolerance_m=1e-6,quadrature_energy_relative_tolerance=1e-4)
    b=dict(sampled_u=np.ones((3,3)),strain=np.ones((3,3,3)),traction=np.ones((3,3)),
           elastic_J=1.,spring_J=1.,force_balance_N=0.,moment_balance_Nm=0.,residual_N=1e-12,minimum_J=.9)
    assert compare(b,b,p)['pass']
    for key in ['strain','traction']:
        a={**b,key:b[key]*1.1}
        assert not compare(a,b,p)['pass']
    a={**b,'sampled_u':b['sampled_u']+2e-6}
    assert compare(a,b,p)['pass']
    assert not compare(a,b,p,quadrature=True)['pass']
    assert not compare({**b,'spring_J':1.02},b,p)['pass']
    assert not compare({**b,'residual_N':1e-8},b,p)['pass']
