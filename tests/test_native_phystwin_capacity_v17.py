import ast
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_capacity import densify_public,geometry_only_auxiliary,ROOT
from scripts.compositional_dynamics_v17.native_phystwin_capacity_fit import configure

def fixture():
    q,_,_=geometry_only_auxiliary();points=(q*[.12,.06,.06]+[0,-.03,-.03]).astype(np.float32)
    data=dict(object_points=np.repeat(points[None],21,axis=0),object_colors=np.full((21,98,3),.5,np.float32),object_visibilities=np.ones((21,98),bool),object_motions_valid=np.ones((20,98),bool),controller_points=np.empty((21,0,3),np.float32),surface_points=np.empty((0,3),np.float32),interior_points=np.zeros((27,3),np.float32))
    return data,dict(total_mass_kg=.27)

def test_auxiliary_geometry_no_extra_motion_and_exact_mass():
    data,metadata=fixture();dense,meta=densify_public(data,metadata)
    for key in ['object_points','object_colors','object_visibilities','object_motions_valid','controller_points']:assert np.array_equal(dense[key],data[key])
    geometry=np.concatenate([dense['object_points'][0],dense['surface_points'],dense['interior_points']])
    assert len(geometry)==729 and len(np.unique(geometry,axis=0))==729
    assert dense['surface_points'].shape==(288,3)and dense['interior_points'].shape==(343,3)
    np.testing.assert_allclose(sum(meta['object_masses_kg']),.27,atol=1e-14)
    data['object_points'][1:]+=123.;mutated,_=densify_public(data,metadata)
    assert np.array_equal(mutated['surface_points'],dense['surface_points'])and np.array_equal(mutated['interior_points'],dense['interior_points'])

def test_capacity_recipe_only_resolution_mass_and_default_radius_change():
    def functions(path):return {x.name:ast.dump(x,include_attributes=False)for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
    original=functions(ROOT/'scripts/compositional_dynamics_v17/native_phystwin_fit.py');capacity=functions(ROOT/'scripts/compositional_dynamics_v17/native_phystwin_capacity_fit.py')
    for key in ['native_data','runtime_classes','run','guard']:assert original[key]==capacity[key]
    class Cfg:
        def load_from_yaml(self,path):self.init_spring_Y=30000.;self.spring_Y_max=100000.;self.dashpot_damping=100.;self.iterations=200;self.drag_damping=3.;self.base_lr=.001
    cfg=Cfg();scale=configure(cfg,dict(object_masses_kg=[.27/729]*729))
    assert cfg.object_radius==.02 and cfg.iterations==200 and cfg.drag_damping==3. and cfg.base_lr==.001
    assert scale==.27/729 and np.isclose(cfg.init_spring_Y,30000*.27/729)
