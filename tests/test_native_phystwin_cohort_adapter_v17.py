import numpy as np
import pytest
from scripts.compositional_dynamics_v17.native_phystwin_cohort_adapter import canonical_points,validate_public,RECIPE
from scripts.compositional_dynamics_v17.native_phystwin_inputs import export_prefix

def fixture():
    surface,interior=canonical_points();tracks=np.repeat(surface[None],21,axis=0);visible=np.ones((21,98),bool);visible[4,10]=False
    data,metadata=export_prefix(initial_tracked_m=surface,observed_prefix_m=tracks,visible_prefix=visible,controller_commands_m=np.empty((21,0,3)),other_surface_m=np.empty((0,3)),interior_m=interior,object_masses_kg=np.full(125,.27/125),frame_dt_s=.01)
    metadata.update(gravity_vector_m_s2=[0,0,-9.8],initial_velocity_m_s=[0,0,0])
    return data,metadata

def test_public_fixed_recipe_and_private_labels_rejected():
    data,metadata=fixture();validate_public(data,metadata)
    assert RECIPE['epochs']==1000 and RECIPE['cma_generations']==100
    with pytest.raises(ValueError,match='labels'):validate_public(data,dict(metadata,material_family='truth'))
    with pytest.raises(ValueError,match='labels'):validate_public(dict(data,future_positions=np.ones((26,98,3))),metadata)

def test_unsanitized_invisible_target_and_correspondence_rejected():
    data,metadata=fixture();data['object_points'][4,10]+=.001
    with pytest.raises(ValueError,match='Invisible'):validate_public(data,metadata)
    data,metadata=fixture();data['interior_points']=data['interior_points'][::-1]
    with pytest.raises(ValueError,match='correspondence'):validate_public(data,metadata)

def test_fit_isolates_native_relative_logs_and_restores_cwd(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts.compositional_dynamics_v17 import native_phystwin_cohort_adapter as adapter
    source=tmp_path/'fake_native.py'
    source.write_text("from pathlib import Path\ndef run(stage):\n    assert stage=='fit'\n    Path('outcmaes').mkdir()\n    Path('outcmaes/cwd.txt').write_text(str(Path.cwd()))\n")
    monkeypatch.setattr(adapter,'ROOT',tmp_path);monkeypatch.setattr(adapter,'SOURCE',source)
    monkeypatch.setattr(adapter.sys,'prefix',str(tmp_path/'runs/compositional-dynamics-v17/render-env'))
    settings=dict(public_directory='public',output_directory='runs/compositional-dynamics-v17/fresh/native/world0')
    monkeypatch.setattr(adapter,'guard',lambda path:settings)
    original=Path.cwd();adapter.fit_world(tmp_path/'world-plan.json')
    private=tmp_path/settings['output_directory']/'native-cwd'
    assert (private/'outcmaes/cwd.txt').read_text()==str(private)
    assert Path.cwd()==original and not(tmp_path/'outcmaes').exists()
