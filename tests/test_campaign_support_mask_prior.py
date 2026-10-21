"""Pure area, canonical-realizer and independent sampler controls; no responses."""
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.campaign_hypothesis_inputs import make_prior,realize_hypothesis
from experiments.campaign_physical_prior import FACE_NORMALS,FACE_BASES
from experiments.campaign_support_mask_prior import clip_halfplane,polygon_area,support_mask_distribution,sample_center_given_mask


@pytest.fixture(scope='module')
def geometry():
    public=json.loads(Path('runs/gaussian-campaign-v01/hypothesis-inputs-v01/public-H0.json').read_text())
    prior=make_prior(public)
    return public,prior,support_mask_distribution(prior)


def test_exact_halfplane_area_conservation():
    square=[(0,0),(1,0),(1,1),(0,1)]
    first=clip_halfplane(square,(1,1),1);second=clip_halfplane(square,(-1,-1),-1)
    assert polygon_area(first)==polygon_area(second)==Fraction(1,2)
    assert polygon_area(first)+polygon_area(second)==polygon_area(square)==1


def test_exact_face_cell_and_merged_node_mass_without_normalization(geometry):
    _,prior,d=geometry
    assert d['total_probability_exact']=='1' and not d['probability_renormalized'] and not d['positive_area_dropped']
    assert sum(Fraction(m['probability_exact']) for m in d['masks'])==1
    assert len({m['node_id'] for m in d['masks']})==len(d['masks'])
    for face in prior.feasible_faces[0]:
        for cell in prior.feasible_cells[0,int(face)]:
            regions=[r for m in d['masks'] for r in m['regions'] if r['face']==face and r['atlas_cell']==cell]
            assert sum(Fraction(r['joint_probability_exact']) for r in regions)==Fraction(1,len(prior.feasible_faces[0])*len(prior.feasible_cells[0,int(face)]))
    for mask in d['masks']:assert sum(Fraction(r['probability_given_mask_exact']) for r in mask['regions'])==1
    with pytest.raises(NotImplementedError):support_mask_distribution(prior,bc=1)


def test_canonical_prior_sample_maps_to_actual_physical_realizer_and_retained_region(geometry):
    public,prior,d=geometry;state=prior.sample(1024,np.random.default_rng(20261107));lookup={m['node_id']:m for m in d['masks']}
    for index in np.flatnonzero(state.boundary_type==0)[:64]:
        spec=realize_hypothesis(public,state.take([index]),prior);node=spec['fixture_nodes'][0]
        regions=[r for r in lookup[node]['regions'] if r['face']==state.support_face[index] and r['atlas_cell']==state.support_cells[index,0]]
        assert regions
        cell=int(state.support_cells[index,0]);xy=np.array([cell//prior.grid,cell%prior.grid])
        uv=-prior.support_uv_limit+(xy+state.coordinates[index,4:6])*(2*prior.support_uv_limit/prior.grid)
        def contains(region):
            p=np.asarray(region['uv_polygon']);a=np.roll(p,-1,axis=0)-p;b=uv-p
            return np.all(a[:,0]*b[:,1]-a[:,1]*b[:,0]>=-1e-15)
        assert any(contains(r) for r in regions)


def test_independent_existing_prior_monte_carlo_frequencies(geometry):
    _,prior,d=geometry;state=prior.sample(200000,np.random.default_rng(20261108));selected=np.flatnonzero(state.boundary_type==0)
    counts=np.zeros(len(prior.reference_points),dtype=int)
    for face_value in prior.feasible_faces[0]:
        face=int(face_value);ids=selected[state.support_face[selected]==face]
        cells=state.support_cells[ids,0];xy=np.stack([cells//prior.grid,cells%prior.grid],1)
        uv=-prior.support_uv_limit+(xy+state.coordinates[ids,4:6])*(2*prior.support_uv_limit/prior.grid)
        # Independent raw-world Euclidean nearest-node assignments, no polygons.
        world=prior.center+FACE_NORMALS[face]*prior.size/2+(uv@FACE_BASES[face])*prior.size
        bound=(prior.minimum if face<3 else prior.maximum)[face%3]
        candidates=np.flatnonzero(np.isclose(prior.reference_points[:,face%3],bound,atol=prior.surface_tolerance_m,rtol=0))
        winner=candidates[np.argmin(np.sum((world[:,None]-prior.reference_points[candidates][None])**2,axis=-1),axis=1)]
        np.add.at(counts,winner,1)
    expected=np.zeros(len(counts))
    for mask in d['masks']:expected[mask['node_id']]=mask['probability']
    observed=counts/len(selected);tolerance=np.maximum(6*np.sqrt(expected*(1-expected)/len(selected)),8/len(selected))
    assert np.all(abs(observed-expected)<=tolerance)
    assert not np.any(counts[expected==0])


def test_conditional_center_draws_preserve_actual_point_mask(geometry):
    _,prior,d=geometry;rng=np.random.default_rng(20261109)
    for mask in d['masks']:
        for sample in sample_center_given_mask(d,mask['node_id'],12,rng):
            face=sample['face'];bound=(prior.minimum if face<3 else prior.maximum)[face%3]
            candidates=np.flatnonzero(np.isclose(prior.reference_points[:,face%3],bound,atol=prior.surface_tolerance_m,rtol=0))
            node=candidates[np.argmin(np.linalg.norm(prior.reference_points[candidates]-sample['world_center_m'],axis=-1))]
            assert node==mask['node_id']
            assert np.all(np.asarray(sample['local_cell_coordinates'])>=0) and np.all(np.asarray(sample['local_cell_coordinates'])<=1)
