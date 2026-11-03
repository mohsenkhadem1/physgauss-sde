import numpy as np
import pytest
from scripts.complex_scenes_v18.contact_metrics import trace,compare
from scripts.complex_scenes_v18.world_queries import Query
from scripts.complex_scenes_v18.pair_contact import material_surface


def test_contact_metric_uses_moving_rigid_state_and_counts_missing_events():
    q,_=material_surface(9);x=q*np.array([.12,.06,.06])+[0,-.03,-.03];positions=np.tile(x,(2,1,1,1));times=np.array([0,.002]);query=Query('rigid_impact',0)
    with pytest.raises(ValueError):trace(query,positions,times)
    reference=trace(query,positions,times,np.array([[[.06,0,.075]],[[.06,0,.04]]]))
    event=reference['events']['body_0:striker'];assert event['first_contact_s']==.002
    assert abs(event['maximum_sampled_penetration_m']-.015)<1e-10
    missed=trace(query,positions,times,np.array([[[.06,0,.075]],[[.06,0,.075]]]))
    assert compare(reference,missed)[0]['missed_contact']
