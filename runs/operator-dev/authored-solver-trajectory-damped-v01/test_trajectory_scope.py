import numpy as np
import pytest
from experiments import campaign_solver_trajectory_damped as trajectory


def test_prefix_bounds_are_separate_predeclared_physical_quantities():
    q=np.zeros((4,3));v=q.copy();R=np.zeros((2,3,3))
    assert trajectory.prefix_error(q,v,R,q,v,R)['passed']
    for channel,value in [('q',2.01e-7),('v',2.41e-5),('R',1e-6)]:
        a,b,c=q.copy(),v.copy(),R.copy()
        {'q':a,'v':b,'R':c}[channel].flat[0]=value
        assert not trajectory.prefix_error(a,b,c,q,v,R)['passed']


def test_run_must_pass_new_protocol_guard_before_model_or_output(monkeypatch):
    def reject(*args):raise ValueError('guard-stop')
    monkeypatch.setattr(trajectory,'validate_protocol',reject)
    with pytest.raises(ValueError,match='guard-stop'):trajectory.run('not-a-real-protocol','0'*64)
