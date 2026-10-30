import numpy as np
import pytest
from scripts.coupled_contact_v13.physics import make_block
from scripts.coupled_contact_v13.surface_spatial_verify import sample,probes,metric,inspect

def test_sorted_fraction_interpolator_matches_saved_tetrahedra_for_nonaffine_field():
    rest,tets=make_block((.08,.06,.05),(4,4,4),(.03,0,.06));rng=np.random.default_rng(812)
    query=rng.uniform(rest.min(0),rest.max(0),(31,3));frames=rng.normal(size=(2,len(rest),3));expected=[]
    for p in query:
        for tet in tets:
            x=rest[tet];b=np.linalg.solve((x[1:]-x[0]).T,p-x[0]);w=np.r_[1-b.sum(),b]
            if min(w)>=-1e-10:expected.append(np.einsum('i,tic->tc',w,frames[:,tet]));break
    assert np.allclose(sample(rest,tets,frames,query),np.array(expected).transpose(1,0,2))
    bad=query.copy();bad[0,0]=rest[:,0].max()+.001
    with pytest.raises(ValueError,match='outside'):sample(rest,tets,frames,bad)
    with pytest.raises(ValueError,match='absent'):sample(rest,tets[:1],frames,query)

def test_common_probes_affine_fields_and_rigid_shape_metric():
    sampled=[]
    for n in (4,6):
        rest,tets=make_block((.08,.06,.05),(n,n,n),(.03,0,.06));sparse,dense=probes(rest)
        assert sparse.shape==(27,3) and dense.shape==(2678,3)
        frames=np.array([rest,rest@np.diag([1.1,.9,1.03])+[.1,.2,.3]])
        sampled.append(sample(rest,tets,frames,dense))
    assert np.allclose(sampled[0],sampled[1]);assert metric(*sampled)['world_rms_m']<1e-15
    shifted=sampled[0]+[1,2,3];m=metric(sampled[0],shifted)
    assert m['world_rms_m']>1 and m['shape_rms_m']<1e-12

def test_missing_companion_is_explicit_unqualified(tmp_path):
    result=inspect(tmp_path,'object','slide',{'gates':{'penetration_m':.001}},lambda *a:1)
    assert not result['qualified'] and result['reasons']

def test_identical_meshes_do_not_qualify_as_spatial_companions(tmp_path):
    import json
    rest,tets=make_block((.08,.06,.05),(2,2,2));frames=np.array([rest,rest]);times=np.array([0.,.01])
    for folder in ('reference','reference-spatial'):
        path=tmp_path/folder/'o'/'slide';path.mkdir(parents=True)
        (path/'complete.json').write_text(json.dumps(dict(success=True,diagnostics=dict(min_j=1.,max_penetration_m=0.,dt=1e-5))))
        np.savez(path/'trajectory.npz',rest=rest,tets=tets,frames=frames,times=times,velocities=np.zeros_like(frames),centers=np.zeros((2,1,3)),reactions=np.zeros((2,1,3)))
    result=inspect(tmp_path,'o','slide',{'gates':{'penetration_m':.001}},lambda *a:1)
    assert not result['qualified'] and any('Identical meshes' in r for r in result['reasons'])
