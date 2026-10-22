import json
import numpy as np
import torch
from experiments.campaign_nonlinear_excitation import (
    build_model,geometry,command,nonrigid_rms,simulate,column_metrics,prepare,sha,affine_patch_check)

torch.set_num_threads(2)


def test_conforming_refinement_domain_material_patch_and_load_preserved():
    meshes=[geometry(i) for i in range(3)]
    volumes=[];areas=[]
    for level,(X,T,F) in enumerate(meshes):
        assert len(T)==2*8**level
        assert np.array_equal(X[:5],meshes[0][0])
        assert len(np.unique(X,axis=0))==len(X)
        det=np.linalg.det(np.swapaxes(X[T[:,1:]]-X[T[:,:1]],-1,-2))
        assert np.all(det>0);volumes.append(det.sum()/6)
        tri=X[F];areas.append(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2)
        # Each refined original triangle covers the same physical boundary.
        model=build_model(mesh_level=level)
        forces=command('balanced_axial',.1,1.)
        np.testing.assert_allclose(model.known_force(forces).sum(0).numpy(),forces.sum(0),rtol=0,atol=1e-15)
    np.testing.assert_allclose(volumes,volumes[0],rtol=1e-14,atol=0)
    np.testing.assert_allclose(areas,areas[0],rtol=1e-14,atol=0)
    patch=affine_patch_check();assert patch['passed']


def test_known_world_programs_and_best_rigid_removal():
    for p in ['balanced_axial','transverse_couple_1','transverse_couple_2']:
        f=command(p,.1,1.);np.testing.assert_array_equal(f.sum(0),np.zeros(3))
        assert np.isclose(np.linalg.norm(f[0]),1.)
        np.testing.assert_array_equal(command(p,-.1,1.),np.zeros((2,3)))
    X=geometry()[0];a=.7;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    assert nonrigid_rms(X,X@R.T+np.array([1.,2.,3.]))<1e-15
    assert nonrigid_rms(X,1.06*X)>.001


def test_exact_implicit_logparameter_tangent_matches_independent_reruns():
    # Only two steps: separate perturbed forward solves, no local fitted surrogate.
    E=5000.;K=2.;args=('balanced_axial',1.,120)
    r,a,_=simulate(E,K,*args,sensitivities=True,duration=2/120)
    assert r['completed']
    eps=1e-4
    for j in (0,1):
        parameters=[E,K];parameters[j]*=np.exp(eps)
        rp,ap,_=simulate(*parameters,*args,duration=2/120)
        parameters=[E,K];parameters[j]*=np.exp(-eps)
        rm,am,_=simulate(*parameters,*args,duration=2/120)
        assert rp['completed'] and rm['completed']
        finite=(ap['positions_m']-am['positions_m'])/(2*eps)
        np.testing.assert_allclose(a['position_logparameter_jacobian_m'][...,j],finite,rtol=2e-5,atol=3e-12)


def test_protocol_budget_gates_and_column_confounding(tmp_path):
    path=tmp_path/'protocol.json';d=prepare(path)
    assert d['pilot']['budget_cases']==12 and d['wall_cap_s']==600
    assert d['mesh']['if_initial_fails_one_continuation_to_tets']==128
    assert d['factorial']['all_cases_retained_even_target_failures']
    assert len(sha(path))==64 and json.loads(path.read_text())==d
    parallel=np.stack([np.arange(1,5),2*np.arange(1,5)],axis=1)
    assert not column_metrics(parallel)['separation_diagnostic_passed']
    assert column_metrics(np.eye(2))['separation_diagnostic_passed']
