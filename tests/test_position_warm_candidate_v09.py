import numpy as np
from scripts.position_ood.smooth_fixture import SmoothDensePositionFEM
from scripts.position_ood.fixture import controls
from scripts.position_ood.inverse import fit,physical
from scripts.phystwin.sequential_fixture import points

def test_alternative_warm_start_is_selected_by_acquired_positions():
    m=SmoothDensePositionFEM(1,'cpu');truth=np.r_[np.log([14000.,4500.,7000.]),.1,-.1,-.8,.2,.1,-.1,.1,0.];poses=np.array([controls(a) for a in ['rest','translate_y','translate_z']]);ids=np.argmin(np.linalg.norm(m.geo.rest[:,None]-points()[None],axis=-1),axis=0)
    observed=np.array([m.predict(physical(m.tensor(truth)),m.tensor(p),1)[ids].numpy() for p in poses]);visible=np.isclose(points()[:,1],-.04);bad=np.r_[np.log([20000.,5000.,5000.]),np.zeros(8)]
    result=fit(m,poses,observed,visible,1,iterations=1,initial=bad,material_field=True,flexible_field=True,alternatives=[truth])
    assert result['loss']<1e-16
    np.testing.assert_allclose(result['z'],truth,atol=1e-8)
