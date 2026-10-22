"""No solves: independently reject changed controls/timestamps/source hashes."""
import numpy as np
import pytest
from experiments import render_nonlinear_mesh_comparison as render
from experiments.campaign_nonlinear_excitation import geometry


def panels():
    arrays=[];summaries=[]
    # Preserve the original solver command's multiplication clock exactly;
    # metadata midpoint arrays use division, as saved by the immutable source.
    force=np.stack([render.command('balanced_axial',(k+.5)*(1/240),1.,.2) for k in range(48)])
    for level in range(3):
        rest,tets,_=geometry(level)
        arrays.append(dict(rest_m=rest,tets=tets,positions_m=np.broadcast_to(rest,(49,*rest.shape)).copy(),times_s=np.arange(49)/240,
            held_force_midpoints_s=(np.arange(48)+.5)/240,held_patch_forces_N=force.copy()))
        summaries.append(dict(completed=True,tetrahedra=len(tets),pattern='balanced_axial',young_Pa=5000.,fixture_scale=2.,amplitude_N_per_patch=1.,hz=240))
    return arrays,summaries


def test_true_shared_controls_times_material_support_and_refined_rest():
    p,s=panels();times=render.validate_panels(p,s)
    assert len(times)==49 and times[-1]==.2
    np.testing.assert_array_equal(p[0]['held_patch_forces_N'].sum(1),0.)
    assert 240/24==10


def test_different_known_controls_rejected_without_render():
    p,s=panels();p[1]['held_patch_forces_N'][23,0,0]+=.01
    with pytest.raises(ValueError,match='original balanced1N'):render.validate_panels(p,s)
    p,s=panels();s[-1]['fixture_scale']=10
    with pytest.raises(ValueError,match='Same nominal'):render.validate_panels(p,s)


def test_different_actual_timestamps_are_not_interpolated_or_relabelled():
    p,s=panels();p[-1]['times_s'][12]+=.0001
    with pytest.raises(ValueError,match='actual240Hz'):render.validate_panels(p,s)


def test_hash_guard_rejects_changed_source_before_archive_array_read(tmp_path,monkeypatch):
    monkeypatch.setattr(render,'ROOT',tmp_path)
    own=tmp_path/'experiments/render_nonlinear_mesh_comparison.py';own.parent.mkdir();own.write_text('synthetic frozen source')
    monkeypatch.setattr(render,'__file__',str(own))
    physics=tmp_path/'physics.json';physics.write_text('{}')
    recipe=dict(schema='unconverged-nonlinear-mesh-video-v0.1',CPU_threads=2,GPU_used=False,
        source_sha256={'experiments/render_nonlinear_mesh_comparison.py':render.sha(own)},physics_protocol=render.record(physics),cases=[])
    path=tmp_path/'recipe.json';render.write(path,recipe);digest=render.sha(path)
    assert render.validate_recipe('recipe.json',digest)['CPU_threads']==2
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No archive reads after invalid source hash'))
    own.write_text('changed')
    with pytest.raises(ValueError,match='changed'):render.validate_recipe('recipe.json',digest)
