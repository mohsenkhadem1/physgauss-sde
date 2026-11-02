import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import native_data,configure

def data():
    x=np.array([[0.,0,.03],[.12,0,.03]],np.float32)
    return dict(object_points=np.repeat(x[None],21,axis=0),object_colors=np.zeros((21,2,3),np.float32),object_visibilities=np.ones((21,2),bool),object_motions_valid=np.ones((20,2),bool),controller_points=np.empty((21,0,3),np.float32),surface_points=np.empty((0,3),np.float32),interior_points=np.array([[.06,0,0]],np.float32))

def test_forecast_ignores_all_prefix_deformation_and_visibility():
    a=data();b={k:v.copy()for k,v in a.items()};b['object_points'][1:]=np.nan;b['object_visibilities'][:]=False
    x=native_data(a,-.07,26);y=native_data(b,-.07,26)
    for key in x:assert np.array_equal(x[key],y[key])
    assert not x['object_visibilities'].any() and not x['object_motions_valid'].any()
    np.testing.assert_allclose(x['object_points'][0,:,2],.10,atol=1e-8)
    assert np.isfinite(x['object_points']).all()

def test_si_config_changes_dimensional_coefficients_not_drag_or_learning():
    class Cfg:
        def load_from_yaml(self,path):
            self.init_spring_Y=30000.;self.spring_Y_max=100000.;self.dashpot_damping=100.;self.drag_damping=3.;self.iterations=200;self.base_lr=.001
    cfg=Cfg();alpha=configure(cfg,dict(object_masses_kg=[.00216]*125))
    assert alpha==.00216 and cfg.iterations==200 and cfg.base_lr==.001 and cfg.drag_damping==3.
    assert np.isclose(cfg.init_spring_Y,64.8)and np.isclose(cfg.dashpot_damping,.216)
    assert cfg.dt*cfg.num_substeps==.01 and cfg.reverse_z is False
