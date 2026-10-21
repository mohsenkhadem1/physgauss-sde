import numpy as np
import pytest
from experiments.campaign_solver_trajectory_audit import prefix_agreement,STATE_KEYS


def arrays():
    source={k:np.zeros((41,2,3)) for k in STATE_KEYS};new={v:np.zeros((61,2,3)) for v in STATE_KEYS.values()}
    source['actual_state_times_s']=np.arange(41)/20;new['actual_state_times_s']=np.arange(61)/20
    last={v:source[k][-1].copy() for k,v in STATE_KEYS.items()};last['time_s']=np.array(2.)
    return new,source,last


def test_prefix_all_physical_arrays_and_last_accepted():
    new,old,last=arrays();assert prefix_agreement(new,old,last)['passed']
    last['positions_m'][0,0]=1e-8
    with pytest.raises(ValueError,match='Last accepted|last accepted'):prefix_agreement(new,old,last)


@pytest.mark.parametrize('field,value',[('positions_m',2.1e-7),('auxiliary_velocities_point_chord_m_s',2.5e-5),('handle_rotations',1e-6)])
def test_predeclared_position_velocity_rotation_limits(field,value):
    new,old,last=arrays();new[field][5,0,0]=value
    assert not prefix_agreement(new,old,last)['passed']


def test_actual_clock_must_match_without_upsampling():
    new,old,last=arrays();new['actual_state_times_s'][5]+=1e-15
    with pytest.raises(ValueError,match='clocks'):prefix_agreement(new,old,last)
