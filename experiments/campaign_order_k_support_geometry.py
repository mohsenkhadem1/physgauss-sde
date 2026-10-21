"""Single-cell exact order-K support geometry prototype, with no responses.

Regions are canonical node SETS. Facet crossings rank exact affine squared
 distances after a bounded rational outward step, handling simultaneous ties.
Area exhaustion is mandatory; incomplete traversal never produces prior weights.
"""
from __future__ import annotations
from collections import deque
from fractions import Fraction as Q
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
try:
    from campaign_support_mask_prior import rational,dot,clip_halfplane,polygon_area,cell_bounds
    from campaign_physical_prior import FACE_BASES,FACE_NORMALS,PriorState
    from campaign_hypothesis_inputs import make_prior,realize_hypothesis
except ImportError:
    from experiments.campaign_support_mask_prior import rational,dot,clip_halfplane,polygon_area,cell_bounds
    from experiments.campaign_physical_prior import FACE_BASES,FACE_NORMALS,PriorState
    from experiments.campaign_hypothesis_inputs import make_prior,realize_hypothesis


def clean_polygon(p):
    result=[]
    for v in p:
        v=tuple(v)
        if not result or v!=result[-1]:result.append(v)
    if len(result)>1 and result[0]==result[-1]:result.pop()
    return result


def affine_scores(points,origin,jacobian):
    """Distance squared minus the common uv-quadratic term, exactly."""
    origin=tuple(map(rational,origin));B=[tuple(map(rational,row)) for row in jacobian]
    result=[]
    for p in points:
        r=tuple(rational(x)-y for x,y in zip(p,origin))
        result.append((tuple(-2*dot(r,[row[j] for row in B]) for j in range(2)),dot(r,r)))
    return result


def selected_set(scores,node_ids,k,uv):
    return tuple(sorted(sorted(range(len(scores)),key=lambda i:(dot(scores[i][0],uv)+scores[i][1],node_ids[i]))[:k]))


def inequalities(scores,selected):
    chosen=set(selected)
    for s in selected:
        for t in range(len(scores)):
            if t not in chosen:
                yield tuple(a-b for a,b in zip(scores[s][0],scores[t][0])),scores[t][1]-scores[s][1]


def exact_region(scores,selected,lower,upper,node_ids=None):
    if node_ids is not None and any(scores[s]==scores[t] and node_ids[s]>node_ids[t] for s in selected for t in range(len(scores)) if t not in selected):return []
    p=[(lower[0],lower[1]),(upper[0],lower[1]),(upper[0],upper[1]),(lower[0],upper[1])]
    for a,b in inequalities(scores,selected):
        p=clean_polygon(clip_halfplane(p,a,b))
        if len(p)<3:return []
    return p


def across_facet(scores,a,b,lower,upper):
    """First open ranking chamber across an interior positive-length facet.

    Bound the step by EVERY nonzero affine distance-order root and box root.
    All zero roots are crossed together, so no single-swap genericity assumed.
    """
    if a==b:return None
    if any(a[j]==b[j] and a[j] in (lower[j],upper[j]) for j in range(2)):return None
    midpoint=tuple((x+y)/2 for x,y in zip(a,b));direction=(b[1]-a[1],a[0]-b[0]);roots=[]
    values=[dot(c,midpoint)+d for c,d in scores];slopes=[dot(c,direction) for c,d in scores]
    for i in range(len(scores)):
        for j in range(i):
            slope=slopes[i]-slopes[j]
            if slope:
                root=-(values[i]-values[j])/slope
                if root>0:roots.append(root)
    for j in range(2):
        if direction[j]:
            for bound in (lower[j],upper[j]):
                root=(bound-midpoint[j])/direction[j]
                if root>0:roots.append(root)
    epsilon=min(roots)/2 if roots else Q(1)
    result=tuple(x+epsilon*d for x,d in zip(midpoint,direction))
    if not all(lower[j]<result[j]<upper[j] for j in range(2)):raise ValueError('Facet crossing escaped cell')
    return result


def enumerate_order_k_cell(scores,node_ids,k,lower,upper,*,maximum_regions=10000):
    """Exact convex order-K partition in ONE rectangle, abort if not certified."""
    if type(k) is not int or not 1<=k<=len(scores) or len(set(node_ids))!=len(node_ids):raise ValueError('Distinct IDs and valid integerK required')
    if type(maximum_regions) is not int or maximum_regions<1:raise ValueError('Positive explicit enumeration bound required')
    scores=[(tuple(map(rational,a)),rational(b)) for a,b in scores];lower=tuple(map(rational,lower));upper=tuple(map(rational,upper))
    if not all(a<b for a,b in zip(lower,upper)):raise ValueError('Positive cell required')
    # A parabola meets each nonidentical affine tie line at at most2 points.
    # Therefore2L+1 distinct rational candidates guarantee a strict seed.
    pairs=[(i,j) for i in range(len(scores)) for j in range(i) if scores[i]!=scores[j]]
    for index in range(1,2*len(pairs)+2):
        t=Q(index,2*len(pairs)+3);seed=(lower[0]+(upper[0]-lower[0])*t,lower[1]+(upper[1]-lower[1])*t*t)
        if all(dot(scores[i][0],seed)+scores[i][1]!=dot(scores[j][0],seed)+scores[j][1] for i,j in pairs):break
    else:raise ValueError('No exact strict seed; unexpected tie geometry')
    todo=deque([selected_set(scores,node_ids,k,seed)]);seen=set();regions=[]
    while todo:
        selected=todo.popleft()
        if selected in seen:continue
        seen.add(selected)
        if len(seen)>maximum_regions:raise RuntimeError('Explicit region budget exceeded; no partial weights')
        p=exact_region(scores,selected,lower,upper,node_ids);area=polygon_area(p)
        if not area:raise ValueError('Zero-area traversal seed/neighbor; explicit degeneracy unsupported, no dropped positive region')
        regions.append(dict(selected=selected,node_ids=tuple(sorted(node_ids[i] for i in selected)),polygon=p,area=area))
        for a,b in zip(p,p[1:]+p[:1]):
            point=across_facet(scores,a,b,lower,upper)
            if point is not None:
                neighbor=selected_set(scores,node_ids,k,point)
                if neighbor==selected:raise ValueError('Unchanged outward ranking; degenerate traversal needs investigation')
                if neighbor not in seen:todo.append(neighbor)
    declared=(upper[0]-lower[0])*(upper[1]-lower[1]);covered=sum((r['area'] for r in regions),Q(0))
    if covered!=declared:raise ValueError('Exact area exhaustion failed; no normalization repair')
    # Independent pairwise intersection certificate, not only summed area.
    for i,r in enumerate(regions):
        for s in regions[:i]:
            intersection=r['polygon']
            for a,b in inequalities(scores,s['selected']):intersection=clip_halfplane(intersection,a,b)
            if polygon_area(intersection):raise ValueError('Positive-area overlap; no repair')
    return dict(regions=regions,declared_area=declared,covered_area=covered,pairwise_positive_overlap_count=0,visited_sets=len(seen),complete=True)


def run_single_cell(public_path,output,*,face=0,cell=0,k=16):
    """One predeclared H0 BC1 face0/cell0 geometry benchmark only."""
    if (face,cell,k)!=(0,0,16):raise ValueError('Single frozen face0/cell0/K16 scope required')
    root=Path(__file__).resolve().parents[1];public_path=root/public_path;out=root/output
    out.mkdir(parents=True,exist_ok=False)
    public=json.loads(public_path.read_text());prior=make_prior(public)
    if face not in prior.feasible_faces[1] or cell not in prior.feasible_cells[1,face]:raise ValueError('Existing BC1 eligibility required')
    sources=['experiments/campaign_order_k_support_geometry.py','experiments/campaign_support_mask_prior.py','experiments/campaign_hypothesis_inputs.py','experiments/campaign_physical_prior.py']
    pins={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
    protocol=dict(schema='single-cell-K16-exact-prototype-v0.1',face=face,atlas_cell=cell,k=k,maximum_regions=10000,source_sha256=pins,public_path=str(public_path.relative_to(root)),public_sha256=hashlib.sha256(public_path.read_bytes()).hexdigest(),CPU_threads=2,physics_GPU_observations=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();bound=(prior.minimum if face<3 else prior.maximum)[face%3]
    nodes=np.flatnonzero(np.isclose(prior.reference_points[:,face%3],bound,atol=prior.surface_tolerance_m,rtol=0));origin=prior.center+FACE_NORMALS[face]*prior.size/2;B=FACE_BASES[face].T*prior.size[:,None]
    scores=affine_scores(prior.reference_points[nodes],origin,B);lower,upper=cell_bounds(prior,cell)
    result=enumerate_order_k_cell(scores,nodes.tolist(),k,lower,upper);elapsed=time.monotonic()-started
    rng=np.random.default_rng(20261110);checked=0
    # Independent current canonical realizer, not the polygon ranking helper.
    for local in rng.uniform(.001,.999,(128,2)):
        x=(prior.lower+prior.upper)/2;x[4:6]=local
        state=PriorState(x[None],np.array([1]),np.array([0]),np.array([face]),np.array([[cell,0]]),np.array([[int(f[0]) for f in prior.incident_faces]]))
        fixture=tuple(realize_hypothesis(public,state,prior)['fixture_nodes']);uv=tuple(a+rational(float(t))*(b-a) for a,b,t in zip(lower,upper,local))
        hits=[]
        for r in result['regions']:
            if all(dot(a,uv)<=b for a,b in inequalities(scores,r['selected'])):hits.append(r['node_ids'])
        if hits!=[fixture]:raise ValueError('Canonical realizer and exact region assignment disagree')
        checked+=1
    exported=[dict(fixture_nodes=list(r['node_ids']),area_uv_exact=str(r['area']),probability_given_cell_exact=str(r['area']/result['declared_area']),uv_polygon_exact=[[str(v) for v in p] for p in r['polygon']]) for r in result['regions']]
    report=dict(completed=True,face=face,atlas_cell=cell,k=k,candidate_nodes=len(nodes),regions=len(exported),enumeration_and_certificate_seconds=elapsed,covered_area_uv_exact=str(result['covered_area']),declared_area_uv_exact=str(result['declared_area']),pairwise_positive_overlap_count=0,realizer_checks=checked,probability_renormalized=False,positive_area_dropped=False,source_sha256=pins,conditional_regions=exported,scope='ONE eligible BC1 cell only; no global K16/K9/moving prior, physics, cache or observations')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--public',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();r=run_single_cell(a.public,a.output);print(json.dumps({k:v for k,v in r.items() if k!='conditional_regions'}))
