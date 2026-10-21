"""Geometry-only point-pin pushforward of the existing normalized atlas prior.

Exact rational polygon clipping preserves all positive area and total mass.
Float64 exports approximate the exact values; no numerical renormalization.
No observations, responses, simulation, material selection or prior changes.
"""
from __future__ import annotations
from fractions import Fraction as Q
import math
import numpy as np
try:
    from campaign_physical_prior import ConditionalPhysicalPrior,FACE_BASES,FACE_NORMALS
except ImportError:
    from experiments.campaign_physical_prior import ConditionalPhysicalPrior,FACE_BASES,FACE_NORMALS


def rational(value):return value if isinstance(value,Q) else Q.from_float(float(value))
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))


def clip_halfplane(polygon,normal,bound):
    """Exact convex intersection normal@uv<=bound; rational vertices returned."""
    p=[tuple(map(rational,v)) for v in polygon];a=tuple(map(rational,normal));c=rational(bound)
    if not p:return []
    result=[];previous=p[-1];old=dot(a,previous)-c
    for current in p:
        new=dot(a,current)-c
        if (new<=0)!=(old<=0):
            fraction=old/(old-new)
            result.append(tuple(x+fraction*(y-x) for x,y in zip(previous,current)))
        if new<=0:result.append(current)
        previous=current;old=new
    return result


def fan_areas(polygon):
    p=[tuple(map(rational,v)) for v in polygon]
    result=[]
    for a,b in zip(p[1:-1],p[2:]):
        x=(a[0]-p[0][0],a[1]-p[0][1]);y=(b[0]-p[0][0],b[1]-p[0][1])
        area=(x[0]*y[1]-x[1]*y[0])/2
        if area<0:raise ValueError('Negative exact convex fan area; no geometry repair')
        result.append(area)
    return result


def polygon_area(polygon):return sum(fan_areas(polygon),Q(0))


def cell_bounds(prior,cell):
    limit=rational(prior.support_uv_limit);width=2*limit/prior.grid
    index=[int(cell)//prior.grid,int(cell)%prior.grid]
    return tuple(-limit+i*width for i in index),tuple(-limit+(i+1)*width for i in index)


def support_mask_distribution(prior,bc=0):
    """p(global node | BC0,H), with exact conditional face/cell/center regions.

    Fractions interpret the declared binary64 geometric parameters as exact
    real values; categorical eligibility is exactly the existing atlas. The
    continuous local-UV prior is unchanged. Sampling/exports use float64.
    """
    if type(prior) is not ConditionalPhysicalPrior:raise TypeError('Existing normalized ConditionalPhysicalPrior required')
    if type(bc) is not int or bc!=0:raise NotImplementedError('Point-pin BC0 only; order-K patches deferred')
    faces=prior.feasible_faces[0];face_weight=Q(1,len(faces));limit=rational(prior.support_uv_limit)
    width=2*limit/prior.grid;cell_area=width**2;masks={};face_records=[]
    for value in faces:
        face=int(value);bound=(prior.minimum if face<3 else prior.maximum)[face%3]
        candidates=np.flatnonzero(np.isclose(prior.reference_points[:,face%3],bound,atol=prior.surface_tolerance_m,rtol=0))
        origin=prior.center+FACE_NORMALS[face]*prior.size/2;B=FACE_BASES[face].T*prior.size[:,None]
        origin_q=tuple(map(rational,origin));Bq=[tuple(map(rational,row)) for row in B]
        relative=[tuple(rational(x)-o for x,o in zip(prior.reference_points[node],origin_q)) for node in candidates]
        square=[(-limit,-limit),(limit,-limit),(limit,limit),(-limit,limit)];voronoi=[]
        for i,s in enumerate(relative):
            region=square.copy()
            for j,t in enumerate(relative):
                if j==i:continue
                if s==t:
                    if candidates[j]<candidates[i]:region=[];break
                    continue
                difference=tuple(y-x for x,y in zip(s,t))
                normal=tuple(2*dot([row[k] for row in Bq],difference) for k in range(2))
                rhs=dot(difference,tuple(x+y for x,y in zip(s,t)))
                region=clip_halfplane(region,normal,rhs)
                if not region:break
            voronoi.append(region)
        cells=prior.feasible_cells[0,face];cell_weight=face_weight/len(cells);cell_records=[]
        for value in cells:
            cell=int(value);lower,upper=cell_bounds(prior,cell);covered=Q(0)
            for node,region in zip(candidates,voronoi):
                if not region:continue
                if any(max(p[k] for p in region)<lower[k] or min(p[k] for p in region)>upper[k] for k in range(2)):continue
                polygon=region
                for normal,rhs in [((1,0),upper[0]),((-1,0),-lower[0]),((0,1),upper[1]),((0,-1),-lower[1])]:
                    polygon=clip_halfplane(polygon,normal,rhs)
                area=polygon_area(polygon);covered+=area
                if area==0:continue  # Exactly zero measure, no positive cutoff.
                probability=cell_weight*area/cell_area
                entry=masks.setdefault(int(node),{'node_id':int(node),'fixture_nodes':[int(node)],'regions':[],'_masses':[]})
                entry['_masses'].append(probability)
                entry['regions'].append(dict(face=face,atlas_cell=cell,
                    uv_polygon=[[float(v) for v in point] for point in polygon],
                    uv_polygon_exact=[[str(v) for v in point] for point in polygon],
                    area_uv=float(area),area_uv_exact=str(area),joint_probability=float(probability),joint_probability_exact=str(probability),
                    cell_lower_uv=[float(v) for v in lower],cell_upper_uv=[float(v) for v in upper]))
            if covered!=cell_area:raise ValueError('Exact uncovered/overlapping eligible cell; no mass normalization')
            cell_records.append(dict(atlas_cell=cell,covered_area_uv_exact=str(covered),declared_area_uv_exact=str(cell_area),
                declared_probability=float(cell_weight),declared_probability_exact=str(cell_weight),signed_area_error_uv=0.))
        face_records.append(dict(face=face,probability=float(face_weight),probability_exact=str(face_weight),eligible_cells=len(cells),
            world_origin_m=origin.tolist(),world_UV_jacobian_m=B.tolist(),cells=cell_records))
    result=[];total=Q(0)
    for node,entry in sorted(masks.items()):
        mass=sum(entry.pop('_masses'),Q(0));total+=mass
        entry.update(probability=float(mass),probability_exact=str(mass))
        for region in entry['regions']:
            conditional=Q(region['joint_probability_exact'])/mass
            region.update(probability_given_mask=float(conditional),probability_given_mask_exact=str(conditional))
        result.append(entry)
    if total!=1:raise ValueError('Exact pushforward mass differs from1; never renormalized')
    floating_sum=math.fsum(m['probability'] for m in result)
    return dict(schema='point-pin-normalized-prior-pushforward-v0.1',boundary_index=0,conditioned_on_BC=True,
        node_count=len(prior.reference_points),constraint_axes=[True,True,True],support_active=[True,False],
        masks=result,faces=face_records,total_probability=float(total),total_probability_exact=str(total),
        float_export_probability_sum=floating_sum,float_export_probability_error=floating_sum-1,
        maximum_absolute_cell_area_error_uv=0.,prior_sha256=prior.metadata()['prior_sha256'],
        probability_renormalized=False,positive_area_dropped=False,
        arithmetic='Exact Fraction clipping/areas from declared binary64 geometry and continuous2limit/grid atlas; float64 export only.',
        metric='Full world Euclidean nearest-face-node affineUV bisectors; stable global-node-ID ties.',
        material_independent=True,response_or_observation_inputs_used=False,
        scope='p(mask|BC0,H); no newcenterprior or four-BC posterior.')


def sample_center_given_mask(distribution,node_id,n,rng):
    """Float64 draws from exact p(face,cell,UV|node), without rejection/redraw."""
    if type(n) is not int or n<1:raise ValueError('Positive sample count required')
    mask=next(m for m in distribution['masks'] if m['node_id']==node_id);regions=mask['regions'];mass=Q(mask['probability_exact'])
    weights=[Q(r['joint_probability_exact']) for r in regions]
    cdf=np.array([float(sum(weights[:i+1],Q(0))/mass) for i in range(len(weights))])
    ids=np.searchsorted(cdf,rng.random(n),side='right');faces={f['face']:f for f in distribution['faces']};samples=[]
    for index in ids:
        region=regions[int(index)];p=[[Q(v) for v in point] for point in region['uv_polygon_exact']]
        areas=fan_areas(p);area=Q(region['area_uv_exact'])
        triangle_cdf=np.array([float(sum(areas[:i+1],Q(0))/area) for i in range(len(areas))])
        triangle=int(np.searchsorted(triangle_cdf,rng.random(),side='right'));vertices=np.asarray(p,dtype=float)
        root=np.sqrt(rng.random());v=rng.random()
        uv=(1-root)*vertices[0]+root*(1-v)*vertices[triangle+1]+root*v*vertices[triangle+2]
        face=faces[region['face']];world=np.asarray(face['world_origin_m'])+np.asarray(face['world_UV_jacobian_m'])@uv
        lower=np.asarray(region['cell_lower_uv']);upper=np.asarray(region['cell_upper_uv'])
        samples.append(dict(face=region['face'],atlas_cell=region['atlas_cell'],uv=uv.tolist(),
            local_cell_coordinates=((uv-lower)/(upper-lower)).tolist(),world_center_m=world.tolist()))
    return samples
