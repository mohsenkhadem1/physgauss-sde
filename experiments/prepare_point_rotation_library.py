"""Prepare/build observation-independent corrected prior ray libraries.

The old archives supply ONLY geometry_ids/physical_coordinates members.
All corrected rays are recomputed; old ranges/likelihoods/weights never enter.
Preparation writes a frozen plan; execution requires a separate explicit run.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
try:
    from campaign_support_basis_cache import load_support_basis_cache
    from campaign_point_rotation_forward import PointRotationForward,signed_tet_diagnostics
    from campaign_point_material_reference import mask_probabilities,prefix_indices
    from campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
    from campaign_sparse_depth_queries import predict_ranges
except ImportError:
    from experiments.campaign_support_basis_cache import load_support_basis_cache
    from experiments.campaign_point_rotation_forward import PointRotationForward,signed_tet_diagnostics
    from experiments.campaign_point_material_reference import mask_probabilities,prefix_indices
    from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
    from experiments.campaign_sparse_depth_queries import predict_ranges

ROOT=Path(__file__).resolve().parents[1]
CACHE=dict(path='runs/gaussian-campaign-v01/point-support-bases-v01/cache-manifest.json',sha256='642f50c696129c0cafda781b87a3342d3effcc25ca5484172afb25b604977664')
PUBLIC='runs/operator-dev/unknown-point-oracle-inputs-v01/public/manifest.json'
SEEDS=[20261112,20261113,20261114]
ARCHIVE_HASHES=['498f56a9fb6ae2c7427c70604d82026f8e0bacd16d30dacba867a7cc34e9656d','d240bcf778d1bba4394079535ec0f50586302c62968985e8847d75eac8cbcf28','cea4be76fa0ad75613145d4fc62859df66f789e9a992e18f559e04defafc5d90']
INPUT_ARCHIVES=[dict(path=f'runs/gaussian-campaign-v01/unknown-point-oracle-v01/joint-reference-v01/reference-{seed}.npz',sha256=h) for seed,h in zip(SEEDS,ARCHIVE_HASHES)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=name:raise ValueError('Canonical repository-relative path required')
    p=(ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Repository escape')
    return p


def checked(record):
    p=resolve(record['path'])
    if sha(p)!=record['sha256']:raise ValueError('Frozen artifact changed '+record['path'])
    return p


def record(path):return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_coordinates_only(path):
    """Strict NPZ member allowlist. No range/visibility/likelihood/weights read."""
    with np.load(path,allow_pickle=False) as archive:
        ids=archive['geometry_ids'].copy()
        physical=archive['physical_coordinates'].copy()
    return ids,physical


def integration_records(ids,physical,probabilities,*,count=1024,prefix=256):
    expected=np.repeat(sorted(probabilities),count)
    if (ids.dtype!=np.int64 or not np.array_equal(ids,expected) or physical.dtype!=np.float64
            or physical.shape!=(len(ids),4) or not np.isfinite(physical).all()):raise ValueError('Exact complete ordered prior coordinate catalogue required; never redraw/drop')
    low=np.array([1500.,400.,.28,.35]);high=np.array([20000.,850.,.42,.85])
    if np.any(physical<low) or np.any(physical>high):raise ValueError('Original declared material-prior coordinates required')
    rows=prefix_indices(ids,prefix,probabilities)
    mass=np.array([probabilities[int(i)]/count for i in ids],dtype=np.float64)
    prefix_mass=np.array([probabilities[int(i)]/prefix for i in ids[rows]],dtype=np.float64)
    if abs(mass.sum()-1)>64*np.finfo(float).eps or abs(prefix_mass.sum()-1)>64*np.finfo(float).eps:raise ValueError('Exact prior mass export inconsistency; never renormalize')
    return dict(prefix_indices_256=rows,prior_mass_1024=mass,prior_mass_256=prefix_mass)


def corrected_range_batch(forward,clock,ids,physical,rest,tets,faces,queries,*,device='cpu'):
    result=forward.predict(clock,ids,physical,device=device)
    # This call order is intentional: sensor receives corrected particle surfaces.
    rays=predict_ranges(result['positions_m'],faces,queries,device=device,particle_chunk=64)
    diagnostics=signed_tet_diagnostics(result['positions_m'],rest,tets)
    return rays,diagnostics


def sources(cache):
    names=['campaign_point_rotation_forward.py','prepare_point_rotation_library.py','campaign_batched_point_rotation.py',
        'campaign_sparse_depth_queries.py','campaign_ray_sensor.py','censored_range_observation.py',
        'render_independent_trajectory.py','campaign_point_material_reference.py']
    pins={**cache['source_sha256'],**{'experiments/'+n:sha(ROOT/'experiments'/n) for n in names}}
    for path,digest in pins.items():
        if sha(resolve(path))!=digest:raise ValueError('Frozen source changed '+path)
    return pins


def prepare(destination,output):
    folder=resolve(destination)
    if folder.exists() or resolve(output).exists():raise ValueError('Fresh prospective plan/output prefixes required')
    cache=read(checked(CACHE));public=read(resolve(PUBLIC));probabilities=mask_probabilities(read(checked(public['point_mask_prior'])))
    if len(probabilities)!=55 or cache['public_hardware']['sha256']!=public['public_hardware']['sha256']:raise ValueError('Exact shared H0/all55 support prior required')
    queries=read(checked(public['public_acquired_queries']))
    if ([q['frame_index'] for q in queries]!=[10,20] or sum(len(q['xs']) for q in queries)!=72
            or public['public_acquired_queries']['sha256']!='9699169b51e02184f6eb41372d4c00dcc8261eda844f8d712b2b2452f2b677e1'):raise ValueError('Frozen original rest-only72 base rays required')
    for r in INPUT_ARCHIVES:checked(r) # Archive digests only; no NPZ members in preparation.
    plan=dict(schema='point-rotation-observation-independent-prior-library-v0.2',status='PREPARED_NOT_GPU_LAUNCH_AUTHORIZATION',
        cache=CACHE,public_manifest=record(resolve(PUBLIC)),public_hardware=public['public_hardware'],support_prior=public['point_mask_prior'],
        public_queries=public['public_acquired_queries'],input_archives=INPUT_ARCHIVES,seeds=SEEDS,
        allowed_old_archive_members=['geometry_ids','physical_coordinates'],forbidden_old_members=['predicted_visible','predicted_residual_m','log_likelihood','weights'],
        counts_per_mask=[256,1024],mask_count=55,particles_per_seed=55*1024,total_particles=3*55*1024,
        prefix_rule='First256 in EACH ordered mask; preserve exact original theta/geometry rows. No redraw/reorder/selection.',
        integration_mass='p(mask)/Nmask for eachcount. Finitefloat export checked, never normalized or thresholded.',
        source_sha256=sources(cache),program_id=PROGRAM_IDS[0],command_manifest=make_exciting_commands(PROGRAM_IDS[0]).public_manifest(),
        output=output,device='cuda:2',CPU_threads=2,particle_chunk=512,ray_particle_chunk=64,
        geometry_diagnostics='Perparticle minimumJ and nonpositive-tet-framecount across ALL61frames/alltets; retain inversions and every priorparticle. No candidatefilter/shapeprojection.',
        sensor_input='PointRotationForward perparticle exp correction BEFORE rayrender; no oldpredictedranges read.',
        output_fields=['geometry_ids','physical_coordinates','predicted_visible','predicted_residual_m','prefix_indices_256','prior_mass_1024','prior_mass_256','minimum_J','nonpositive_tet_frame_count'],
        observations_or_cohort_truth_used=False,likelihood_or_posterior_computed=False,
        approximation='Candidate geometry-only correction to frozen linear-modal mechanics, not correctednonlinearforces/handleorientations/dynamics/calibratedmodel.',
        failure_policy='Preserve partialoutputs/status, stop on nonfinite/invalidinput/sourcechange; no retry/redraw/drop. FiniteJ<=0 remainsdiagnostic with rays retained.')
    folder.mkdir(parents=True);path=folder/'library-plan.json';write(path,plan)
    return record(path)


def validate_plan(plan_record):
    plan=read(checked(plan_record));cache=read(checked(plan['cache']))
    if (plan['schema']!='point-rotation-observation-independent-prior-library-v0.2'
            or plan['status']!='PREPARED_NOT_GPU_LAUNCH_AUTHORIZATION' or plan['cache']!=CACHE
            or plan['input_archives']!=INPUT_ARCHIVES or plan['seeds']!=SEEDS or plan['counts_per_mask']!=[256,1024]
            or plan['source_sha256']!=sources(cache) or plan['device']!='cuda:2' or plan['CPU_threads']!=2
            or plan['particle_chunk']!=512 or plan['particles_per_seed']!=55*1024
            or plan['allowed_old_archive_members']!=['geometry_ids','physical_coordinates']):raise ValueError('Frozen exact corrected-library scope required')
    for key in ['public_manifest','public_hardware','support_prior','public_queries']:checked(plan[key])
    return plan


def run(plan_record):
    plan=validate_plan(plan_record)
    if any(os.environ.get(n)!='2' for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    torch.set_num_threads(2);out=resolve(plan['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(plan=plan_record,source_sha256=plan['source_sha256'],started_with_external_root_authorization=True))
    status=dict(pid=os.getpid(),terminal=False,completed=False,processed_particles=0,source_sha256=plan['source_sha256']);write(out/'execution.json',status)
    started=time.monotonic();records=[]
    try:
        packed,public,ids,distribution=load_support_basis_cache(plan['cache']['path'],plan['cache']['sha256'])
        if public!=read(checked(plan['public_hardware'])) or len(ids)!=55:raise ValueError('Frozen shared public H0/cache differs')
        probability=mask_probabilities(distribution);queries=read(checked(plan['public_queries']))
        rest=torch.tensor(public['rest_points_m'],dtype=torch.float64,device=plan['device']);tets=public['tetrahedra'];faces=public['surface_faces']
        forward=PointRotationForward(packed,rest);clock=make_exciting_commands(plan['program_id'])
        if clock.public_manifest()!=plan['command_manifest']:raise ValueError('Known command changed')
        for seed,archive in zip(plan['seeds'],plan['input_archives']):
            geometry,physical=load_coordinates_only(checked(archive));mass=integration_records(geometry,physical,probability)
            count=len(geometry);visible=np.empty((count,72),dtype=bool);ranges=np.empty((count,72),dtype=np.float64)
            minimum=np.empty(count,dtype=np.float64);inversions=np.empty(count,dtype=np.int64)
            for lo in range(0,count,512):
                if any(sha(resolve(p))!=h for p,h in plan['source_sha256'].items()):raise ValueError('Frozen source changed duringlibrarybuild')
                with torch.no_grad():
                    rays,diagnostic=corrected_range_batch(forward,clock,torch.tensor(geometry[lo:lo+512]),torch.tensor(physical[lo:lo+512]),rest,tets,faces,queries,device=plan['device'])
                hi=lo+len(rays.visible);visible[lo:hi]=rays.visible;ranges[lo:hi]=rays.residual_m
                minimum[lo:hi]=diagnostic['minimum_J'].cpu().numpy();inversions[lo:hi]=diagnostic['nonpositive_tet_frame_count'].cpu().numpy()
                status.update(seed=seed,processed_particles=status['processed_particles']+hi-lo,elapsed_s=time.monotonic()-started)
                write(out/'execution.json',status)
            path=out/f'corrected-prior-{seed}.npz';np.savez_compressed(path,geometry_ids=geometry,physical_coordinates=physical,
                predicted_visible=visible,predicted_residual_m=ranges,minimum_J=minimum,nonpositive_tet_frame_count=inversions,**mass)
            records.append(dict(seed=seed,library=record(path),particles=count,minimum_J=float(minimum.min()),
                particles_with_nonpositive_J=int((inversions>0).sum()),nonpositive_tet_frame_count=int(inversions.sum()),
                geometry_array_sha256=hashlib.sha256(geometry.tobytes()).hexdigest(),physical_array_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),categories_dropped=[]))
            write(out/'completed-seeds.json',records)
        summary=dict(status='COMPLETED_ALL3_CORRECTED_OBSERVATION_INDEPENDENT_PRIOR_LIBRARIES',plan=plan_record,records=records,
            source_sha256=plan['source_sha256'],elapsed_s=time.monotonic()-started,old_archive_members_read=plan['allowed_old_archive_members'],
            old_ray_or_likelihood_or_posterior_members_read=False,cohort_or_truth_used=False,posterior_computed=False)
        write(out/'summary.json',summary);status.update(terminal=True,completed=True,elapsed_s=time.monotonic()-started);write(out/'execution.json',status)
        return summary
    except BaseException as error:
        status.update(terminal=True,completed=False,error=repr(error),elapsed_s=time.monotonic()-started);write(out/'execution.json',status);write(out/'failure.json',status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--destination',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    result=prepare(args.destination,args.output) if args.action=='prepare' else run(dict(path=args.plan,sha256=args.plan_sha256));print(json.dumps(result))
