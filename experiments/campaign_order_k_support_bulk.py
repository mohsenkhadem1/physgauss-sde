"""Bounded geometry-only H0 K16 / individual K9 prior pushforwards.

Sequential CPU2, hard600s wall limit; incomplete results remain explicitly
unqualified. Opposite-patch joint catalogue and moving laws are not expanded.
"""
from __future__ import annotations
from fractions import Fraction as Q
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
try:
    from campaign_order_k_support_geometry import affine_scores,enumerate_order_k_cell
    from campaign_support_mask_prior import cell_bounds
    from campaign_physical_prior import FACE_BASES,FACE_NORMALS,PriorState
    from campaign_hypothesis_inputs import make_prior,realize_hypothesis
except ImportError:
    from experiments.campaign_order_k_support_geometry import affine_scores,enumerate_order_k_cell
    from experiments.campaign_support_mask_prior import cell_bounds
    from experiments.campaign_physical_prior import FACE_BASES,FACE_NORMALS,PriorState
    from experiments.campaign_hypothesis_inputs import make_prior,realize_hypothesis
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def opposite_patch_joint_mass(face_count,first_given_face,second_given_opposite):
    """First-face1/F; deterministic opposite, then independent conditionalUVs."""
    if type(face_count) is not int or face_count<1:raise ValueError('Positive first-face count')
    a,b=Q(first_given_face),Q(second_given_opposite)
    if not 0<=a<=1 or not 0<=b<=1:raise ValueError('Conditional probabilities required')
    return a*b/face_count


def cell_region_weights(areas,cell_area,eligible_cells):
    areas=list(map(Q,areas));cell_area=Q(cell_area)
    if type(eligible_cells) is not int or eligible_cells<1 or any(a<=0 for a in areas) or sum(areas,Q(0))!=cell_area:
        raise ValueError('All positive areas and exact coverage required; no repair')
    return [a/cell_area/eligible_cells for a in areas]


def specs(prior):
    # fixed/moving atlas equality is tested, never inferred from sameK alone.
    if list(prior.feasible_faces[1])!=list(prior.feasible_faces[3]):raise ValueError('K16 atlases differ')
    plan=[]
    for bc,k in [(1,16),(2,9)]:
        for f in prior.feasible_faces[bc]:
            f=int(f)
            if bc==1 and not np.array_equal(prior.feasible_cells[1,f],prior.feasible_cells[3,f]):raise ValueError('K16 cells differ')
            for c in prior.feasible_cells[bc,f]:plan.append(dict(k=k,bc=bc,face=f,atlas_cell=int(c)))
    return plan


def run(public_path,output):
    if any(os.environ.get(x)!='2' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 thread environment required')
    p=Path(public_path)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Relative public path required')
    out=ROOT/output;out.mkdir(parents=True,exist_ok=False);(out/'cells').mkdir()
    public=json.loads((ROOT/p).read_text());prior=make_prior(public);plan=specs(prior)
    if public['id']!='H0':raise ValueError('Exact H0 scope required')
    source_names=['experiments/campaign_order_k_support_bulk.py','experiments/campaign_order_k_support_geometry.py','experiments/campaign_support_mask_prior.py','experiments/campaign_physical_prior.py','experiments/campaign_hypothesis_inputs.py']
    pins={p:sha(ROOT/p) for p in source_names}
    protocol=dict(schema='bounded-H0-order-K-geometry-v0.1',source_sha256=pins,public_path=str(p),public_sha256=sha(ROOT/p),prior_sha256=prior.metadata()['prior_sha256'],CPU_threads=2,maximum_wall_seconds=600,maximum_regions_per_cell=10000,plan=plan,joint_catalogue_expanded=False,physics_cache_GPU_observations=False)
    dump(out/'protocol.json',protocol)
    records=[];start=time.monotonic();current=None;failure=None;completed=False
    def timed_out(signum,frame):raise TimeoutError('Prospective600s geometry budget exhausted; incomplete catalogue unqualified')
    old_handler=signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,600)
    dump(out/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False,planned_cells=len(plan),completed_cells=0))
    try:
        for current in plan:
            before=time.monotonic();face=current['face'];k=current['k'];bc=current['bc'];cell=current['atlas_cell'];bound=(prior.minimum if face<3 else prior.maximum)[face%3]
            nodes=np.flatnonzero(np.isclose(prior.reference_points[:,face%3],bound,atol=prior.surface_tolerance_m,rtol=0))
            origin=prior.center+FACE_NORMALS[face]*prior.size/2;B=FACE_BASES[face].T*prior.size[:,None]
            scores=affine_scores(prior.reference_points[nodes],origin,B);lower,upper=cell_bounds(prior,cell)
            result=enumerate_order_k_cell(scores,nodes.tolist(),k,lower,upper,maximum_regions=10000)
            conditional=cell_region_weights([r['area'] for r in result['regions']],result['declared_area'],len(prior.feasible_cells[bc,face]))
            regions=[]
            for r,mass in zip(result['regions'],conditional):
                # Rational polygon mean is strict interior of a convex positive region.
                uv=[sum((v[j] for v in r['polygon']),Q(0))/len(r['polygon']) for j in range(2)]
                local=np.array([float((u-a)/(b-a)) for u,a,b in zip(uv,lower,upper)])
                x=(prior.lower+prior.upper)/2;x[4:6]=local
                state=PriorState(x[None],np.array([bc]),np.array([0]),np.array([face]),np.array([[cell,int(prior.feasible_cells[bc,(face+3)%6][0]) if bc==2 else 0]]),np.array([[int(f[0]) for f in prior.incident_faces]]))
                actual=realize_hypothesis(public,state,prior)['support_node_axis_masks'][0]
                selected=tuple(np.flatnonzero(np.any(np.asarray(actual),axis=-1)))
                if selected!=r['node_ids']:raise ValueError('Current canonical realizer differs from exact region mask')
                regions.append(dict(fixture_nodes=list(r['node_ids']),uv_polygon_exact=[[str(v) for v in u] for u in r['polygon']],area_uv_exact=str(r['area']),probability_given_face_exact=str(mass)))
            target=out/'cells'/f'K{k}-face{face}-cell{cell}.json'
            dump(target,dict(**current,regions=regions,area_exact=str(result['declared_area']),coverage_exact=True,positive_overlap_count=0))
            rec=dict(**current,path=str(target.relative_to(ROOT)),sha256=sha(target),regions=len(regions),candidate_nodes=len(nodes),elapsed_seconds=time.monotonic()-before,covered_area_uv_exact=str(result['covered_area']),declared_area_uv_exact=str(result['declared_area']),pairwise_positive_overlap_count=0,canonical_realizer_checks=len(regions))
            records.append(rec)
            dump(out/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False,planned_cells=len(plan),completed_cells=len(records),elapsed_seconds=time.monotonic()-start,current=current))
        # Exact conditional face mass, then uniform FIRST-face mass. No joint expansion.
        catalogues={}
        for bc,k in [(1,16),(2,9)]:
            F=len(prior.feasible_faces[bc]);faces={};merged={}
            for f in prior.feasible_faces[bc]:
                face=int(f);rows=[r for r in records if r['k']==k and r['face']==face];entries={}
                for row in rows:
                    for region in json.loads((ROOT/row['path']).read_text())['regions']:
                        key=tuple(region['fixture_nodes']);mass=Q(region['probability_given_face_exact']);entries[key]=entries.get(key,Q(0))+mass
                        # K16 mechanical mask merge retains latentface/cell region nuisance.
                        if k==16:merged[key]=merged.get(key,Q(0))+mass/F
                if sum(entries.values(),Q(0))!=1:raise ValueError('Exact face conditional mass failed')
                faces[str(face)]=dict(probability_first_face_exact=str(Q(1,F)),opposite_face=(face+3)%6,eligible_cells=len(rows),conditional_mass_exact='1',masks=[dict(fixture_nodes=list(ids),probability_given_face_exact=str(m)) for ids,m in sorted(entries.items())])
            if k==16 and sum(merged.values(),Q(0))!=1:raise ValueError('K16 exact total mass failed')
            catalogues[str(k)]=dict(faces=faces,merged_masks=[dict(fixture_nodes=list(ids),probability_exact=str(m)) for ids,m in sorted(merged.items())],moving_law_not_collapsed=True if k==16 else None,joint_opposite_pairs_not_expanded=True if k==9 else None)
        dump(out/'geometry-catalogues.json',catalogues);completed=True
    except BaseException as error:
        failure=dict(error=type(error).__name__,message=str(error),current=current,traceback=traceback.format_exc(limit=6));dump(out/'failure.json',failure)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        summary=dict(completed=completed,pid=os.getpid(),terminal=True,planned_cells=len(plan),completed_cells=len(records),elapsed_seconds=time.monotonic()-start,failure=failure,cell_certificates=records,source_sha256=pins,all_positive_regions_retained=True,probability_renormalized=False,joint_opposite_pairs_expanded=False,physics_cache_GPU_observations=False)
        if completed:summary['catalogue']=dict(path=str((out/'geometry-catalogues.json').relative_to(ROOT)),sha256=sha(out/'geometry-catalogues.json'))
        dump(out/'summary.json',summary);dump(out/'execution.json',{k:v for k,v in summary.items() if k not in ['cell_certificates','source_sha256']})
    if not completed:raise RuntimeError('Bounded geometry construction incomplete; preserved, no retry or qualification')
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--public',required=True);a.add_argument('--output',required=True);p=a.parse_args();r=run(p.public,p.output);print(json.dumps({k:v for k,v in r.items() if k!='cell_certificates'}))
