import pytest
from scripts.compositional_dynamics_v17.native_phystwin_replay import validate_setup

def test_reset_clock_and_known_inputs_only():
    s=dict(duration_s=.2,frame_dt_s=.01,floor_z_m=-.04,initial_velocity_m_s=[0,0,0],initial_geometry_same=True,gravity_vector_m_s2=[0,0,-9.8])
    assert validate_setup(s)==21
    assert validate_setup(dict(s,duration_s=.25,floor_z_m=-.07))==26
    with pytest.raises(ValueError):validate_setup(dict(s,hidden_stiffness=10000))
    with pytest.raises(ValueError):validate_setup(dict(s,duration_s=.201))
    with pytest.raises(ValueError):validate_setup(dict(s,gravity_vector_m_s2=[0,0,-9.81]))
