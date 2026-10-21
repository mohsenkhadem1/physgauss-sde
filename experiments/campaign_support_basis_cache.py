"""Reproducible nominal bases for every exact point-support prior category.

Only public H0, an exact geometric prior table, and deterministic nominal
queries are consumed. No observations, finite trajectories or posterior.
Raw .pt files contain geometry tensors only, not dense full systems/live clocks.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from fractions import Fraction as Q
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch
try:
    from campaign_hypothesis_inputs import (known_hardware_from_public,make_prior,realize_hypothesis,
        validate_realized_hypothesis,physical_hardware_digest)
    from campaign_physical_prior import PriorState
    from campaign_support_mask_prior import cell_bounds
    from campaign_linear_hypothesis import LinearHypothesisDynamics
    from campaign_modal_hypothesis import FixedModalLinearDynamics
    from campaign_modal_static_lift import complete_static_lift
    from campaign_packed_boundary_forward import PackedBoundaryMaterialForward,modal_tensor_record
    from campaign_hypothesis_dynamics import runtime_sources
except ImportError:
    from experiments.campaign_hypothesis_inputs import (known_hardware_from_public,make_prior,realize_hypothesis,
        validate_realized_hypothesis,physical_hardware_digest)
    from experiments.campaign_physical_prior import PriorState
    from experiments.campaign_support_mask_prior import cell_bounds
    from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
    from experiments.campaign_modal_hypothesis import FixedModalLinearDynamics
    from experiments.campaign_modal_static_lift import complete_static_lift
    from experiments.campaign_packed_boundary_forward import PackedBoundaryMaterialForward,modal_tensor_record
    from experiments.campaign_hypothesis_dynamics import runtime_sources

ROOT=Path(__file__).resolve().parents[1]
NOMINAL=dict(youngs_modulus_Pa=5000.,density_kg_m3=625.,poissons_ratio=.35,body_linear_drag_s_inverse=.6)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(path):raise ValueError('Canonical repo-relative path required')
    result=(ROOT/p).resolve()
    if not result.is_relative_to(ROOT):raise ValueError('Repository escape')
    return result


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def record(path):return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))
def checked(r):
    p=resolve(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('Frozen artifact changed '+r['path'])
    return p


def sources():
    names=['campaign_support_basis_cache.py','campaign_packed_boundary_forward.py',
        'campaign_support_mask_prior.py','campaign_linear_hypothesis.py','campaign_modal_hypothesis.py',
        'campaign_modal_static_lift.py','campaign_modal_material_forward.py']
    return {**runtime_sources(),**{'experiments/'+n:sha(ROOT/'experiments'/n) for n in names}}


def polygon_centroid_exact(vertices):
    p=[[Q(x) for x in v] for v in vertices];cross=[]
    for a,b in zip(p,p[1:]+p[:1]):cross.append(a[0]*b[1]-b[0]*a[1])
    area2=sum(cross,Q(0))
    if area2<=0:raise ValueError('Positive exact polygon area required')
    return tuple(sum(((a[j]+b[j])*c for a,b,c in zip(p,p[1:]+p[:1],cross)),Q(0))/(3*area2) for j in [0,1])


def representative_spec(public,prior,mask):
    region=min(mask['regions'],key=lambda r:(-Q(r['joint_probability_exact']),r['face'],r['atlas_cell']))
    uv=polygon_centroid_exact(region['uv_polygon_exact']);lower,upper=cell_bounds(prior,region['atlas_cell'])
    local=[float((v-a)/(b-a)) for v,a,b in zip(uv,lower,upper)]
    x=(prior.lower+prior.upper)/2
    x[:4]=[np.log(5000.),np.log(625.),.35,np.log(.6)];x[4:6]=local
    state=PriorState(x[None],np.array([0]),np.array([0]),np.array([region['face']]),
        np.array([[region['atlas_cell'],0]]),np.array([[int(f[0]) for f in prior.incident_faces]]))
    spec=realize_hypothesis(public,state,prior)
    if spec['fixture_nodes']!=[mask['node_id']]:raise ValueError('Actual decoder representative disagrees with exact point category; no redraw/drop')
    # Logs exponentiate to binary64 nominal values, as in the frozen decoder.
    if any(abs(spec['material'][k]-v)>8*np.finfo(float).eps*abs(v) for k,v in NOMINAL.items()):raise ValueError('Nominal material realization changed')
    return spec,dict(selection='Largest EXACT joint-mass conditional region; tie face/cell, exact area centroid',
        face=region['face'],atlas_cell=region['atlas_cell'],centroid_uv_exact=[str(v) for v in uv],
        local_cell_coordinates=local,selected_region_joint_probability_exact=region['joint_probability_exact'],
        query_not_prior_truth_sample=True)


def prepare(public_path,distribution_path,output):
    public=read(resolve(public_path));distribution=read(resolve(distribution_path));prior=make_prior(public)
    masks=distribution['masks'];ids=[m['node_id'] for m in masks]
    if (len(ids)!=55 or len(set(ids))!=55 or sum((Q(m['probability_exact']) for m in masks),Q(0))!=1
            or distribution['prior_sha256']!=prior.metadata()['prior_sha256']
            or distribution['probability_renormalized'] or distribution['positive_area_dropped']):
        raise ValueError('Reviewed exact55-category prior with unchanged unit mass required')
    destination=resolve(output)
    if destination.exists():raise ValueError('Fresh basis-cache directory required')
    destination.mkdir(parents=True);(destination/'specs').mkdir();rows=[]
    for mask in masks:
        spec,representative=representative_spec(public,prior,mask)
        path=destination/'specs'/('node'+str(mask['node_id'])+'.json');dump(path,spec)
        rows.append(dict(node_id=mask['node_id'],probability_exact=mask['probability_exact'],probability=mask['probability'],
            representative=representative,spec=record(path)))
    protocol=dict(schema='all55-nominal-point-support-bases-v0.1',status='FROZEN_ROOT_AUTHORIZED_CPU_CONSTRUCTION',
        public_hardware=record(resolve(public_path)),distribution=record(resolve(distribution_path)),
        nominal_material=NOMINAL,vibration_modes=96,completion='frozen canonical translation completion, at most3 extra modes',
        source_sha256=sources(),output=output,categories=rows,maximum_CPU_workers=4,threads_per_worker=2,GPU_used=False,
        preserve_all_categories=True,category_weights_changed=False,finite_or_observation_inputs=False)
    path=destination/'construction-protocol.json';dump(path,protocol)
    return record(path)


def validate_protocol(path,digest):
    protocol=read(checked(dict(path=path,sha256=digest)))
    if (protocol['schema']!='all55-nominal-point-support-bases-v0.1'
            or protocol['status']!='FROZEN_ROOT_AUTHORIZED_CPU_CONSTRUCTION' or protocol['source_sha256']!=sources()
            or protocol['nominal_material']!=NOMINAL or protocol['vibration_modes']!=96
            or protocol['maximum_CPU_workers']!=4 or protocol['threads_per_worker']!=2):raise ValueError('Frozen basis scope/source mismatch')
    distribution=read(checked(protocol['distribution']));checked(protocol['public_hardware'])
    if [(m['node_id'],m['probability_exact']) for m in distribution['masks']]!=[(r['node_id'],r['probability_exact']) for r in protocol['categories']]:raise ValueError('Category catalogue or exact weights changed')
    for row in protocol['categories']:checked(row['spec'])
    return protocol


def build_one(path,digest,node_id):
    torch.set_num_threads(2);protocol=validate_protocol(path,digest);started=time.monotonic()
    row=next(r for r in protocol['categories'] if r['node_id']==node_id)
    public=read(checked(protocol['public_hardware']));known=known_hardware_from_public(public);spec=read(checked(row['spec']))
    validate_realized_hypothesis(known,spec)
    if spec['fixture_nodes']!=[node_id]:raise ValueError('Actual decoder fixture changed')
    full=LinearHypothesisDynamics.from_hypothesis(known,spec)
    modal=complete_static_lift(FixedModalLinearDynamics(full,96))
    payload=modal_tensor_record(modal,node_id,physical_hardware_digest(known))
    target=resolve(protocol['output'])/('node'+str(node_id)+'.pt')
    if target.exists():raise ValueError('Existing basis retained, no implicit retry')
    torch.save(payload,target)
    return dict(node_id=node_id,cache=record(target),cache_bytes=target.stat().st_size,
        mode_count=modal.mode_count,reduced_coordinate_count=full.size,basis_diagnostics=modal.basis_diagnostics,
        elapsed_s=time.monotonic()-started,worker_pid=os.getpid())


def build(path,digest):
    protocol=validate_protocol(path,digest);destination=resolve(protocol['output']);started=time.monotonic()
    if (destination/'execution.json').exists():raise ValueError('Construction already attempted; preserve it, no retry')
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 thread environment required')
    status=dict(pid=os.getpid(),terminal=False,completed=False,completed_categories=0,planned_categories=55)
    dump(destination/'execution.json',status);rows=[];failures=[];todo=iter(protocol['categories']);live={};stopped=False
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        def submit():
            try:row=next(todo)
            except StopIteration:return False
            future=pool.submit(build_one,path,digest,row['node_id']);live[future]=row['node_id'];return True
        for _ in range(4):submit()
        while live:
            done,_=wait(live,return_when=FIRST_COMPLETED)
            for future in done:
                node=live.pop(future)
                try:rows.append(future.result())
                except BaseException as error:
                    stopped=True;failures.append(dict(node_id=node,error=type(error).__name__,message=str(error)))
            status.update(completed_categories=len(rows),failures=failures,elapsed_s=time.monotonic()-started)
            dump(destination/'execution.json',status)
            if not stopped:
                while len(live)<4 and submit():pass
    completed=not failures and len(rows)==55
    manifest=dict(schema='all55-point-support-basis-cache-v0.1',completed=completed,
        construction_protocol=dict(path=path,sha256=digest),source_sha256=protocol['source_sha256'],
        public_hardware=protocol['public_hardware'],distribution=protocol['distribution'],
        categories=sorted(rows,key=lambda r:r['node_id']),failures=failures,elapsed_s=time.monotonic()-started,
        raw_tensor_cache_LOCAL=True,observations_or_truth_used=False)
    dump(destination/'cache-manifest.json',manifest)
    status.update(terminal=True,completed=completed,elapsed_s=manifest['elapsed_s']);dump(destination/'execution.json',status)
    if not completed:raise RuntimeError('Incomplete category cache retained; no category dropping/retry')
    return record(destination/'cache-manifest.json')


def load_support_basis_cache(manifest_path,manifest_sha256):
    """Return packedforward, publicH0dict, exactglobalIDs, distributiondict."""
    manifest=read(checked(dict(path=manifest_path,sha256=manifest_sha256)))
    if not manifest['completed'] or manifest['source_sha256']!=sources():raise ValueError('Complete frozen-source cache required')
    protocol=validate_protocol(**dict(path=manifest['construction_protocol']['path'],digest=manifest['construction_protocol']['sha256']))
    public=read(checked(manifest['public_hardware']));distribution=read(checked(manifest['distribution']))
    ids=tuple(r['node_id'] for r in protocol['categories'])
    if tuple(r['node_id'] for r in manifest['categories'])!=ids or len(ids)!=55:raise ValueError('All exact55 categories required')
    records={row['node_id']:torch.load(checked(row['cache']),map_location='cpu',weights_only=True) for row in manifest['categories']}
    return PackedBoundaryMaterialForward.from_tensor_records(records,public),public,ids,distribution


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--public',required=True);p.add_argument('--distribution',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('build');p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args()
    result=prepare(args.public,args.distribution,args.output) if args.action=='prepare' else build(args.protocol,args.protocol_sha256)
    print(json.dumps(result),flush=True)
