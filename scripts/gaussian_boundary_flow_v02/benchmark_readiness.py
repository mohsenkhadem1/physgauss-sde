"""Prospective prefix-only numerical/model-adequacy screen of eight input packages.

No evaluator import or future object target access. Four predetermined latent
worlds per case are diagnostic probes, not fitted predictions or benchmark wins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.object_backend_revision import ObjectBoundaryBackendRevision

ROOT=Path(__file__).resolve().parents[2]
CASES=('single_lift_sloth','single_push_sloth','single_lift_zebra','double_lift_zebra',
       'object-soft-task1-demo0','object-soft-task2-demo0','spatial-soft-task1-demo0','spatial-soft-task2-demo0')
SEEDS=(913,914,915)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def mesh_diagnostic(backend):
    model=backend.model;rest=model.rest.detach().cpu().numpy();elements=model.elements.detach().cpu().numpy()
    raw=Delaunay(rest).simplices
    used=np.unique(elements);missing=np.setdiff1d(np.arange(len(rest)),used)
    row=np.repeat(elements,4,axis=1).reshape(-1)
    col=np.tile(elements,(1,4)).reshape(-1)
    graph=coo_matrix((np.ones(len(row)),(row,col)),shape=(len(rest),len(rest))).tocsr()
    groups,labels=connected_components(graph,directed=False)
    dm=np.stack([rest[elements[:,i]]-rest[elements[:,0]]for i in (1,2,3)],axis=-1)
    cond=np.linalg.cond(dm);finite=cond[np.isfinite(cond)]
    return dict(nodes=len(rest),raw_delaunay_tetrahedra=len(raw),retained_tetrahedra=len(elements),
                removed_degenerate_tetrahedra=len(raw)-len(elements),disconnected_node_ids=missing.tolist(),
                connected_components=int(groups),component_sizes=np.bincount(labels).tolist(),
                zero_mass_nodes=np.flatnonzero(model.mass.detach().cpu().numpy()<=0).tolist(),
                tetrahedron_condition_number_max=float(finite.max())if len(finite)else None,
                tetrahedron_condition_number_median=float(np.median(finite))if len(finite)else None,
                nonfinite_condition_numbers=int((~np.isfinite(cond)).sum()),
                mesh_filter=model.mesh_filter,reference_volume_m3=float(model.vol.sum().item()*model.scale**3),
                initial_support_points=len(backend.data.get('support_geometry',backend.data['nodes'])))


def rigid_shape_error(predicted,observed):
    if len(predicted)<3:return None
    p=predicted-predicted.mean(0);q=observed-observed.mean(0)
    U,_,Vh=np.linalg.svd(p.T@q);D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vh)
    aligned=p@(U@D@Vh)+observed.mean(0)
    return float(np.sqrt(np.mean((aligned-observed)**2))*1000)


def prefix_metrics(payload,positions,frames):
    positions=np.asarray(positions);frames=np.asarray(frames,int)
    if not len(frames):return dict(frames_scored=0,coordinate_rmse_mm=None,aligned_shape_rmse_mm=None,per_frame=[])
    initial=payload['observed_initial'];A=payload['observation_matrix']
    projected=initial[None]+np.einsum('pn,tnc->tpc',A,positions-payload['nodes'])
    rows=[]
    for i,t in enumerate(frames):
        mask=payload['observed_mask'][t]&np.isfinite(payload['observed'][t]).all(-1)
        if not np.isfinite(projected[i]).all():
            rows.append(dict(frame=int(t),status='nonfinite_prediction'));continue
        if not mask.any():rows.append(dict(frame=int(t),status='no_valid_observation'));continue
        p=projected[i,mask];q=payload['observed'][t,mask]
        rows.append(dict(frame=int(t),status='scored',points=int(mask.sum()),
                         coordinate_rmse_mm=float(np.sqrt(np.mean((p-q)**2))*1000),
                         mean_l2_mm=float(np.linalg.norm(p-q,axis=-1).mean()*1000),
                         aligned_shape_rmse_mm=rigid_shape_error(p,q)))
    good=[r for r in rows if r['status']=='scored']
    mean=lambda name:float(np.mean([r[name]for r in good if r[name]is not None]))if any(r[name]is not None for r in good)else None
    output=dict(frames_scored=len(good),coordinate_rmse_mm=mean('coordinate_rmse_mm'),
                aligned_shape_rmse_mm=mean('aligned_shape_rmse_mm'),mean_l2_mm=mean('mean_l2_mm'),per_frame=rows,
                interpretation='Raw equilibrium predictions; no frame-zero subtraction, alignment correction or fitting; shape metric removes best rigid transform only for scoring')
    if 0 in frames:
        i=int(np.flatnonzero(frames==0)[0]);displacement=positions[i]-payload['nodes']
        output['raw_initial_node_coordinate_mismatch_mm']=float(np.sqrt(np.mean(displacement**2))*1000)
        output['raw_initial_node_max_distance_mm']=float(np.linalg.norm(displacement,axis=-1).max()*1000)
        output['raw_initial_observation_coordinate_mismatch_mm']=next((r['coordinate_rmse_mm']for r in rows if r['frame']==0 and r['status']=='scored'),None)
    return output


def draw_screen(backend,latent,frames,maxiter,timeout_seconds):
    """Instrument solve results without changing equations, iterates or acceptance."""
    trace=[];positions={};failure=[];original=backend.model.solve;requested=set(frames.tolist())
    def observed_solve(theta,frame,initial=None,**kwargs):
        try:result=original(theta,frame,initial,**kwargs)
        except Exception as error:
            failure.append(dict(frame=int(frame),type=type(error).__name__,message=str(error)));raise
        state,_,diag=result
        trace.append(dict(frame=int(frame),**diag))
        if frame in requested:positions[int(frame)]=state.detach().cpu().numpy()*backend.model.scale+backend.model.origin
        return result
    def expired(*unused):raise TimeoutError('Per-draw readiness time budget exceeded')
    backend.model.solve=observed_solve
    prior_handler=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,timeout_seconds);start=time.monotonic();success=False;error=None
    try:
        output,diags=backend.rollout(latent,frames,maxiter=maxiter)
        if not np.isfinite(output).all():raise RuntimeError('Nonfinite returned coordinates')
        success=True
    except Exception as e:error=dict(type=type(e).__name__,message=str(e))
    finally:
        elapsed=time.monotonic()-start;signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,prior_handler);backend.model.solve=original
    available=np.array(sorted(positions),dtype=int)
    x=np.stack([positions[int(t)]for t in available])if len(available)else np.empty((0,len(backend.data['nodes']),3))
    result=dict(completed=success,seconds=elapsed,error=error,failure_frame=failure[0]['frame']if failure else None,
                requested_frames=frames.tolist(),returned_frames=available.tolist(),canonical_frames_completed=len(trace),
                minimum_J=min(r['minJ']for r in trace)if trace else None,
                maximum_residual=max(r['residual']for r in trace)if trace else None,
                total_iterations=sum(r['iterations']for r in trace),
                prefix_metrics=prefix_metrics(backend.data,x,available),
                partial_metrics_only=not success,diagnostics=trace)
    result['numerical_acceptance_only']=bool(success and result['minimum_J']>.03 and result['maximum_residual']<=1e-5)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:2');parser.add_argument('--maxiter',type=int,default=60)
    parser.add_argument('--per-draw-seconds',type=float,default=300.);parser.add_argument('--input-root',default='runs/gaussian-boundary-flow-v02-inputs')
    parser.add_argument('--output',default='reports/gaussian-boundary-flow-v02/readiness');args=parser.parse_args()
    if args.device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Only authorized GPU0-2 or CPU')
    torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'scripts/gaussian_boundary_flow_v02/object_backend_revision.py',ROOT/'scripts/gaussian_boundary_flow_v02/object_backend.py',ROOT/'scripts/external_benchmark/fem.py',ROOT/'scripts/gaussian_boundary_flow_v02/export_inputs.py']
    plan=dict(scope='Prospective model-adequacy screen using historical prefix observations only; no parameter fitting, evaluator or future target access',
              cases=list(CASES),device=args.device,floor_mode='inferred',draws='neutral logits plus three independent standard-normal logit draws',prior_seeds=list(SEEDS),
              requested_frames='12 equally spaced indices across complete observed prefix; canonical solver visits EVERY intermediate command frame',
              maxiter=args.maxiter,per_draw_seconds=args.per_draw_seconds,
              inputs={case:sha(Path(args.input_root)/case/'manifest.json')for case in CASES},
              source_sha256={str(p.relative_to(ROOT)):sha(p)for p in sources})
    if(out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=plan:raise ValueError('Readiness protocol changed; use a new output directory')
    else:write(out/'protocol.json',plan)
    protocol_sha=sha(out/'protocol.json');cases=[]
    for case in CASES:
        payload=load_package(Path(args.input_root)/case);frames=np.unique(np.linspace(0,payload['train_end']-1,12,dtype=int))
        caseout=out/case;caseout.mkdir(exist_ok=True);init_start=time.monotonic()
        try:
            backend=ObjectBoundaryBackendRevision(payload,args.device,floor_mode='inferred')
            mesh=mesh_diagnostic(backend);init_seconds=time.monotonic()-init_start
        except Exception as error:
            row=dict(case=case,construction_completed=False,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-init_start,protocol_sha256=protocol_sha)
            write(caseout/'summary.json',row);cases.append(row);print(case,'construction FAILED',row['error'],flush=True);continue
        latents=[np.zeros(backend.parameter_dim)]+[np.random.default_rng(seed).normal(size=backend.parameter_dim)for seed in SEEDS]
        rows=[]
        for index,latent in enumerate(latents):
            path=caseout/f'draw-{index}.json'
            if path.exists():
                row=json.loads(path.read_text())
                if row['protocol_sha256']!=protocol_sha or row['latent']!=latent.tolist():raise ValueError('Changed readiness checkpoint')
            else:
                row=draw_screen(backend,latent,frames,args.maxiter,args.per_draw_seconds)
                row.update(case=case,draw=index,seed=None if index==0 else SEEDS[index-1],latent=latent.tolist(),parameters=backend.names,
                           predictor_digest=predictor_digest(payload),protocol_sha256=protocol_sha)
                write(path,row)
            rows.append(row);print(case,'draw',index,'completed',row['completed'],'prefix_mm',row['prefix_metrics']['coordinate_rmse_mm'],'seconds',row['seconds'],flush=True)
        case_summary=dict(case=case,dataset=payload['dataset'],construction_completed=True,construction_seconds=init_seconds,mesh=mesh,
                          observed_frames=payload['train_end'],observed_points=payload['observed'].shape[1],requested_frames=frames.tolist(),
                          successful_draws=sum(r['completed']for r in rows),draws=rows,protocol_sha256=protocol_sha)
        write(caseout/'summary.json',case_summary);cases.append(case_summary)
        write(out/'progress.json',dict(completed_cases=len(cases),total_cases=len(CASES),last_case=case))
    summary=dict(protocol_sha256=protocol_sha,cases=cases,completed_cases=len(cases),total_cases=len(CASES),
                 source_hashes_unchanged=all(sha(ROOT/p)==digest for p,digest in plan['source_sha256'].items()),
                 forecast_target_access=False,fresh_case_access=False,
                 interpretation='Numerical acceptance is not physical adequacy. Prefix errors and initial-shape mismatch are diagnostics only; no held performance or calibrated posterior claim.')
    write(out/'summary.json',summary)

if __name__=='__main__':main()
