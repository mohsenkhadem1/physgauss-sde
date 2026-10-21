from fractions import Fraction as Q
from itertools import combinations
import pytest
from experiments.campaign_order_k_support_geometry import affine_scores,exact_region,enumerate_order_k_cell,selected_set
from experiments.campaign_support_mask_prior import polygon_area


def brute(scores,ids,k):
    result={}
    for selected in combinations(range(len(ids)),k):
        p=exact_region(scores,selected,(Q(-1),Q(-1)),(Q(1),Q(1)),ids)
        a=polygon_area(p)
        if a:result[tuple(sorted(ids[i] for i in selected))]=a
    return result


@pytest.mark.parametrize('k',[1,2,3,4])
def test_exact_traversal_matches_exhaustive_toy_sets(k):
    # Rational asymmetric sites plus symmetric simultaneous bisectors.
    points=[[-.6,-.4,0],[.6,-.4,0],[.6,.4,0],[-.6,.4,0],[0,0,0]]
    scores=affine_scores(points,[0,0,0],[[1,0],[0,1],[0,0]])
    r=enumerate_order_k_cell(scores,[9,3,7,1,12],k,(-1,-1),(1,1))
    assert {x['node_ids']:x['area'] for x in r['regions']}==brute(scores,[9,3,7,1,12],k)
    assert r['covered_area']==r['declared_area']==4
    assert r['pairwise_positive_overlap_count']==0


def test_coincident_sites_stable_ids_and_canonical_unordered_sets():
    scores=affine_scores([[0,0,0],[0,0,0],[1,0,0],[-1,0,0]],[0,0,0],[[1,0],[0,1],[0,0]])
    r=enumerate_order_k_cell(scores,[5,2,8,1],2,(-1,-1),(1,1))
    assert {x['node_ids']:x['area'] for x in r['regions']}==brute(scores,[5,2,8,1],2)
    assert selected_set(scores,[5,2,8,1],1,(Q(0),Q(0)))==(1,)


def test_bounded_failure_never_exports_partial_weights():
    scores=affine_scores([[-.5,0,0],[.5,0,0]],[0,0,0],[[1,0],[0,1],[0,0]])
    with pytest.raises(RuntimeError,match='budget'):enumerate_order_k_cell(scores,[0,1],1,(-1,-1),(1,1),maximum_regions=1)
    with pytest.raises(ValueError):enumerate_order_k_cell(scores,[0,0],1,(-1,-1),(1,1))
