import json
import numpy as np
from scripts.coupled_contact_v13.physics import make_block
from scripts.coupled_contact_v13.surface_spatial_quality import inspect


def put(base,directory,cells,shift=0.,body=0.):
    rest,tets=make_block(size=(.08,.06,.05),cells=(cells,)*3)
    frames=np.stack([rest,rest@np.diag([1.1,.95,1.02])])
    frames[1,:,0]+=shift
    centers=np.zeros((2,1,3));centers[1,0,0]=body
    path=base/directory/'obj'/'curved';path.mkdir(parents=True,exist_ok=True)
    np.savez(path/'trajectory.npz',rest=rest,tets=tets,frames=frames,centers=centers,velocities=np.zeros_like(frames),reactions=np.zeros_like(centers),times=[0.,.6])
    (path/'complete.json').write_text(json.dumps(dict(success=True,diagnostics=dict(min_j=.9,max_penetration_m=0.,dt=25e-6))))
    return path


def test_affine_two_mesh_interpolation_and_separate_body_gate(tmp_path):
    put(tmp_path,'reference',2);put(tmp_path,'reference-spatial',4)
    q=inspect(tmp_path,'obj','curved')
    assert q['qualified'] and q['dense2678']['points']==2678
    assert q['common27']['world_rms_m']<1e-14 and q['dense2678']['world_rms_m']<1e-14
    put(tmp_path,'reference-spatial',4,body=.002)
    q=inspect(tmp_path,'obj','curved')
    assert not q['qualified'] and any('body' in r for r in q['reasons'])


def test_world_gate_retains_rigid_displacement_and_missing_companion(tmp_path):
    put(tmp_path,'reference',2)
    assert not inspect(tmp_path,'obj','curved')['qualified']
    put(tmp_path,'reference-spatial',4,shift=.003)
    q=inspect(tmp_path,'obj','curved')
    assert not q['qualified']
    assert q['common27']['world_rms_m']>.001 and q['dense2678']['world_rms_m']>.001
    assert q['dense2678']['shape_rms_m']<1e-14


def test_invalid_volume_disqualifies_even_identical_trajectories(tmp_path):
    put(tmp_path,'reference',2);p=put(tmp_path,'reference-spatial',4)
    status=json.loads((p/'complete.json').read_text());status['diagnostics']['min_j']=float('nan')
    (p/'complete.json').write_text(json.dumps(status))
    q=inspect(tmp_path,'obj','curved')
    assert not q['qualified'] and any('volume' in r for r in q['reasons'])


def test_actual_step_mismatch_with_identical_saved_times_disqualifies(tmp_path):
    put(tmp_path,'reference',2);p=put(tmp_path,'reference-spatial',4)
    status=json.loads((p/'complete.json').read_text());status['diagnostics']['dt']=12.5e-6
    (p/'complete.json').write_text(json.dumps(status))
    q=inspect(tmp_path,'obj','curved')
    assert not q['qualified'] and any('diagnostic dt' in r for r in q['reasons'])


def test_nonfinite_velocity_or_reaction_disqualifies(tmp_path):
    put(tmp_path,'reference',2)
    for key in ('velocities','reactions'):
        p=put(tmp_path,'reference-spatial',4)
        with np.load(p/'trajectory.npz') as a:data={k:a[k].copy() for k in a.files}
        data[key].flat[-1]=np.nan
        np.savez(p/'trajectory.npz',**data)
        q=inspect(tmp_path,'obj','curved')
        assert not q['qualified'] and any('nonfinite' in r for r in q['reasons'])


def test_identical_saved_mesh_is_not_spatial_qualification(tmp_path):
    put(tmp_path,'reference',2);put(tmp_path,'reference-spatial',2)
    q=inspect(tmp_path,'obj','curved')
    assert not q['qualified'] and any('identical' in r for r in q['reasons'])
