"""Retrospective all32 population deformation audit; no inference or labels.

Proper weighted rigid fits separate bulk motion from shape change. Compatible
linear-tet F supplies objective principal stretches/Green-Lagrange strain.
XYZ displacement summaries and a scalar range-noise scale are not range SNR.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(name):raise ValueError('Canonical repository-relative path required')
    target=(ROOT/p).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Repository escape')
    return target

def checked(rec):
    p=resolve(rec['path'])
    if sha(p)!=rec['sha256']:raise ValueError('Frozen audit input changed: '+rec['path'])
    return p

def record(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def reference_geometry(rest,tets):
    rest=np.asarray(rest,dtype=np.float64);tets=np.asarray(tets)
    if (rest.ndim!=2 or rest.shape[1]!=3 or not np.isfinite(rest).all() or tets.ndim!=2 or tets.shape[1]!=4
        or tets.dtype.kind not in 'iu' or not len(tets) or (tets<0).any() or (tets>=len(rest)).any()):raise ValueError('Finite rest mesh and valid tetrahedra required')
    edge=(rest[tets[:,1:]]-rest[tets[:,0,None]]).transpose(0,2,1);det=np.linalg.det(edge)
    if np.any(det<=0):raise ValueError('Positive oriented reference tetrahedra required')
    volume=det/6;nodal=np.zeros(len(rest))
    for j in range(4):np.add.at(nodal,tets[:,j],volume/4)
    if np.any(nodal<=0):raise ValueError('Every node must have positive lumped reference volume')
    return np.linalg.inv(edge),volume,nodal/nodal.sum()


def rigid_fit(rest,positions,weights):
    """Best proper rotation+translation per frame, with normalized nodal weights."""
    r=np.asarray(rest,dtype=np.float64);q=np.asarray(positions,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if (r.ndim!=2 or r.shape[1]!=3 or q.ndim!=3 or q.shape[1:]!=r.shape or not len(q)
        or w.shape!=(len(r),) or np.any(w<0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1.,rtol=0,atol=1e-14)
        or not np.isfinite(q).all() or not np.isfinite(r).all()):raise ValueError('Finite geometry and normalized nonnegative weights required')
    rc=w@r;qc=np.einsum('n,tnc->tc',w,q);a=r-rc;b=q-qc[:,None]
    H=np.einsum('ni,n,tnj->tij',a,w,b);U,s,Vt=np.linalg.svd(H)
    correction=np.ones((len(q),3));correction[:,-1]=np.where(np.linalg.det(U@Vt)<0,-1.,1.)
    row_rotation=(U*correction[:,None])@Vt;fit=a[None]@row_rotation+qc[:,None]
    total=np.einsum('n,tnc,tnc->t',w,q-r,q-r)/3
    residual=q-fit;nonrigid=np.einsum('n,tnc,tnc->t',w,residual,residual)/3
    return dict(total_coordinate_MSE_m2=total,nonrigid_coordinate_MSE_m2=nonrigid,
        rotation_world_from_rest=row_rotation.transpose(0,2,1),translation_m=qc-np.einsum('i,tij->tj',rc,row_rotation),
        proper_rotation_determinant=np.linalg.det(row_rotation))


def tet_strain(rest,tets,positions):
    inverse,volume,_=reference_geometry(rest,tets);q=np.asarray(positions,dtype=np.float64);tets=np.asarray(tets)
    if q.ndim!=3 or q.shape[1:]!=np.asarray(rest).shape or not np.isfinite(q).all():raise ValueError('Finite frame/node positions required')
    edges=(q[:,tets[:,1:]]-q[:,tets[:,0,None]]).transpose(0,1,3,2)
    F=edges@inverse;J=np.linalg.det(F)
    if np.any(J<=0):raise ValueError('Nonpositive captured tet determinant retained as failure')
    stretches=np.linalg.svd(F,compute_uv=False);deviation=stretches-1
    # Eigenvalues of E=.5(F.T F-I) are .5*(stretch^2-1).
    E_principal=.5*(stretches**2-1);w=volume/volume.sum()
    return dict(principal_stretch_MSE_deviation=np.einsum('e,tec,tec->t',w,deviation,deviation)/3,
        green_lagrange_frobenius_MSE=np.einsum('e,tec,tec->t',w,E_principal,E_principal),
        maximum_absolute_principal_stretch_deviation=np.max(np.abs(deviation),axis=(1,2)),
        minimum_principal_stretch=np.min(stretches,axis=(1,2)),maximum_principal_stretch=np.max(stretches,axis=(1,2)),
        minimum_J=np.min(J,axis=1),maximum_J=np.max(J,axis=1))


def window_metrics(fit,strain,indices):
    ids=np.asarray(indices,dtype=int);total=float(np.sqrt(np.mean(fit['total_coordinate_MSE_m2'][ids])));residual=float(np.sqrt(np.mean(fit['nonrigid_coordinate_MSE_m2'][ids])))
    return dict(frames=len(ids),total_coordinate_RMS_m=total,nonrigid_coordinate_RMS_m=residual,
        nonrigid_to_total_RMS_ratio=residual/total if total>0 else None,
        nonrigid_to_total_squared_motion_fraction=(residual/total)**2 if total>0 else None,
        principal_stretch_RMS_deviation=float(np.sqrt(np.mean(strain['principal_stretch_MSE_deviation'][ids]))),
        green_lagrange_frobenius_RMS=float(np.sqrt(np.mean(strain['green_lagrange_frobenius_MSE'][ids]))),
        maximum_absolute_principal_stretch_deviation=float(np.max(strain['maximum_absolute_principal_stretch_deviation'][ids])),
        minimum_principal_stretch=float(np.min(strain['minimum_principal_stretch'][ids])),maximum_principal_stretch=float(np.max(strain['maximum_principal_stretch'][ids])),
        minimum_J=float(np.min(strain['minimum_J'][ids])),maximum_J=float(np.max(strain['maximum_J'][ids])))


def combine_windows(a,b):
    n=a['frames']+b['frames'];result=dict(frames=n)
    for key in ['total_coordinate_RMS_m','nonrigid_coordinate_RMS_m','principal_stretch_RMS_deviation','green_lagrange_frobenius_RMS']:
        result[key]=float(np.sqrt((a['frames']*a[key]**2+b['frames']*b[key]**2)/n))
    t=result['total_coordinate_RMS_m'];r=result['nonrigid_coordinate_RMS_m'];result.update(nonrigid_to_total_RMS_ratio=r/t if t>0 else None,nonrigid_to_total_squared_motion_fraction=(r/t)**2 if t>0 else None)
    for key in ['maximum_absolute_principal_stretch_deviation','maximum_principal_stretch','maximum_J']:result[key]=max(a[key],b[key])
    for key in ['minimum_principal_stretch','minimum_J']:result[key]=min(a[key],b[key])
    return result


def population_summary(objects):
    if [o['object_index'] for o in objects]!=list(range(16,48)):raise ValueError('All32 paired complete objects required')
    output={};rng=np.random.default_rng(2026120800);bootstrap_ids=rng.integers(0,32,size=(10000,32))
    for weighting in ['uniform_nodal','rest_volume_lumped_mass']:
        output[weighting]={}
        for window in ['A_full','A_common_future','B_full_reset','AB_full','AB_common']:
            rows=[o['windows'][weighting][window] for o in objects];stats={}
            for key in ['total_coordinate_RMS_m','nonrigid_coordinate_RMS_m','nonrigid_to_total_RMS_ratio','nonrigid_to_total_squared_motion_fraction','principal_stretch_RMS_deviation','green_lagrange_frobenius_RMS','maximum_absolute_principal_stretch_deviation']:
                x=np.asarray([r[key] for r in rows],dtype=float)
                if not np.isfinite(x).all():raise ValueError('Undefined population denominator retained; no object filtering')
                stats[key]=dict(mean=float(x.mean()),median=float(np.median(x)),quantile05=float(np.quantile(x,.05)),quantile95=float(np.quantile(x,.95)),minimum=float(x.min()),maximum=float(x.max()),
                    object_bootstrap_mean_95_interval=np.quantile(x[bootstrap_ids].mean(1),[.025,.975]).tolist())
                if key in ['total_coordinate_RMS_m','nonrigid_coordinate_RMS_m']:stats[key]['pooled_coordinate_RMS_m']=float(np.sqrt(np.mean(x*x)))
            stats['minimum_J_all_objects']=min(r['minimum_J'] for r in rows);stats['maximum_J_all_objects']=max(r['maximum_J'] for r in rows)
            stats['minimum_principal_stretch_all_objects']=min(r['minimum_principal_stretch'] for r in rows);stats['maximum_principal_stretch_all_objects']=max(r['maximum_principal_stretch'] for r in rows)
            stats['objects_nonrigid_XYZ_coordinate_RMS_above_0p2mm_scalar_reference']=sum(r['nonrigid_coordinate_RMS_m']>.0002 for r in rows)
            output[weighting][window]=stats
    return output


def prepare(path,execution_record,primary_record,output_prefix):
    """Metadata only: trajectory archive members stay unopened until run()."""
    target=resolve(path)
    if target.exists():raise ValueError('Existing audit protocol retained')
    execution=read(checked(execution_record));primary=read(checked(primary_record));plan_record=execution['cohort_plan'];plan=read(checked(plan_record))
    if (primary.get('schema')!='point-rotation-reserved-view-evaluation-v0.2' or primary.get('completed') is not True or primary.get('objects')!=32
        or primary.get('execution_protocol_sha256')!=execution_record['sha256'] or len(primary['reserved_results'])!=32):raise ValueError('Sealed fresh32 primary required')
    objects=[o for o in plan['objects'] if o['role']=='RESERVED_EVALUATION']
    if [o['index'] for o in objects]!=list(range(16,48)):raise ValueError('Exact32 reserved objects required')
    cases=[]
    for obj,result_record in zip(objects,primary['reserved_results']):
        # Hash result bytes only; material/response-derived values are not parsed.
        expected=resolve(execution['analysis_output'])/'objects'/obj['object_id']/'result.json'
        if checked(result_record)!=expected:raise ValueError('Exact same sealed object IDs required')
        for program in 'AB':
            cell=next(c for c in plan['cells'] if c['id']==obj['object_id']+'__'+program);directory=resolve(cell['output'])
            summary=read(directory/'summary.json');status=read(directory/'execution.json')
            if summary.get('completed') is not True or summary.get('frames')!=61 or status.get('terminal') is not True or status.get('completed') is not True:raise ValueError('All64 finite episodes complete before declaration')
            cases.append(dict(object_index=obj['index'],program=program,trajectory=dict(path=cell['output']+'/trajectory.npz',sha256=summary['trajectory_sha256']),
                summary=record(directory/'summary.json'),execution=record(directory/'execution.json'),protocol=cell['protocol']))
    p=dict(schema='point-rotation-population-deformation-audit-v0.1',status='FROZEN_RETROSPECTIVE_NO_NETWORK_OR_NEW_PHYSICS',
        source_sha256={'experiments/audit_point_rotation_deformation.py':sha(__file__)},execution_protocol=execution_record,sealed_primary=primary_record,cohort_plan=plan_record,
        public_hardware=plan['public_hardware'],cases=cases,output_prefix=output_prefix,CPU_threads=2,GPU=False,wall_limit_s=300,
        choices=dict(uniform_nodal=True,reference_lumped_volume_mass=True,rigid_fit='Weighted least squares proper Kabsch SO(3) + free translation at each saved frame',
            coordinate_RMS='sqrt(mean_frame sum_node normalized_weight * squared_vector_displacement / 3)',ratio='Nonrigid coordinateRMS / total coordinateRMS; exact zero denominator undefined, no epsilon repair',
            compatible_F='Current tet edge matrix times inverse rest edge matrix',strain='Volume-weighted RMS principal-stretch deviation; Green-Lagrange Frobenius RMS; endpoint extrema',
            windows=dict(A_full=list(range(61)),A_common_future=list(range(31,61)),B_full_reset=list(range(61))),
            combined_windows='A+B frame-count-weighted squared quantities, keeping both programs within one independent object',bootstrap_replicates=10000,bootstrap_seed=2026120800,bootstrap_unit='Complete object with paired A/B',
            scalar_range_noise_reference_m=.0002,noise_interpretation='XYZ residual units only; not rendered range SNR, detectability or sensor-space information'),
        no_material_or_support_labels_needed=True,raw_trajectories_copied=False,per_object_statistics_LOCAL=True)
    target.parent.mkdir(parents=True,exist_ok=True);write(target,p);return record(target)


def run(protocol_path,digest):
    p=read(checked(dict(path=protocol_path,sha256=digest)))
    if p.get('schema')!='point-rotation-population-deformation-audit-v0.1' or p.get('CPU_threads')!=2 or p.get('GPU') is not False:raise ValueError('Exact frozen CPU audit required')
    for name,value in p['source_sha256'].items():checked(dict(path=name,sha256=value))
    if p['source_sha256'].get(Path(__file__).relative_to(ROOT).as_posix())!=sha(__file__):raise ValueError('Own source must be frozen')
    for key in ['execution_protocol','sealed_primary','cohort_plan']:checked(p[key])
    if any(os.environ.get(k)!='2' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    output=resolve(p['output_prefix'])
    if output.exists():raise ValueError('Existing diagnostic retained; no overwrite')
    output.mkdir(parents=True);start=time.monotonic();status=dict(terminal=False,completed=False,completed_objects=[],private_labels_read=False,trajectory_arrays_opened=False);write(output/'execution.json',status);objects=[]
    try:
        public=read(checked(p['public_hardware']));rest=np.asarray(public['rest_points_m'],dtype=np.float64);tets=np.asarray(public['tetrahedra'],dtype=np.int64)
        _,volume,mass_weights=reference_geometry(rest,tets);weights=dict(uniform_nodal=np.full(len(rest),1/len(rest)),rest_volume_lumped_mass=mass_weights)
        for i in range(16,48):
            if time.monotonic()-start>p['wall_limit_s']:raise TimeoutError('Bounded300s audit wall limit')
            obj=dict(object_index=i,windows={k:{} for k in weights},per_frame={},source_trajectories={})
            for program in 'AB':
                cell=next(c for c in p['cases'] if c['object_index']==i and c['program']==program)
                for key in ['summary','execution','protocol']:checked(cell[key])
                status['trajectory_arrays_opened']=True
                with np.load(checked(cell['trajectory']),allow_pickle=False) as z:
                    q=z['positions_m'].copy();times=z['times_s'].copy()
                if q.dtype!=np.float64 or q.shape!=(61,len(rest),3) or not np.array_equal(times,np.arange(61)/20):raise ValueError('All61 float64 saved positions/actual20Hz output timestamps required')
                strain=tet_strain(rest,tets,q);obj['per_frame'][program]=dict(times_s=times.tolist(),strain={k:v.tolist() for k,v in strain.items()},weights={});obj['source_trajectories'][program]=cell['trajectory']
                for name,w in weights.items():
                    fit=rigid_fit(rest,q,w);obj['per_frame'][program]['weights'][name]={k:v.tolist() for k,v in fit.items()}
                    if program=='A':
                        obj['windows'][name]['A_full']=window_metrics(fit,strain,range(61));obj['windows'][name]['A_common_future']=window_metrics(fit,strain,range(31,61))
                    else:obj['windows'][name]['B_full_reset']=window_metrics(fit,strain,range(61))
            for name in weights:
                obj['windows'][name]['AB_full']=combine_windows(obj['windows'][name]['A_full'],obj['windows'][name]['B_full_reset'])
                obj['windows'][name]['AB_common']=combine_windows(obj['windows'][name]['A_common_future'],obj['windows'][name]['B_full_reset'])
            objects.append(obj);status['completed_objects'].append(i);status['elapsed_s']=time.monotonic()-start;write(output/'execution.json',status)
        write(output/'per-object-statistics-LOCAL.json',objects)
        summary=dict(schema='point-rotation-population-deformation-summary-v0.1',completed=True,objects=32,programs=['A','B'],frames_per_program=61,
            protocol=record(resolve(protocol_path)),sealed_primary=p['sealed_primary'],public_hardware=p['public_hardware'],tetrahedra=len(tets),nodes=len(rest),rest_volume_m3=float(volume.sum()),
            population=population_summary(objects),units=dict(coordinate_RMS='m',ratios='dimensionless',principal_stretches='dimensionless',Green_Lagrange_strain='dimensionless'),
            independent_unit='Complete object, preserving paired A/B programs',bootstrap=dict(replicates=10000,seed=2026120800),
            mass_weighting='Reference volume lumped one-quarter per incident tet; constant homogeneous density cancels on normalization. No material labels used.',
            rigid_fit_scope='Best proper rigid transform removes rotation and translation independently for each frame; nonrigid residual includes shape change of all scales, not only one constitutive mode.',
            range_noise_comparison='0.2mm is a scalar foreground range-noise reference. XYZ coordinate residuals differ from directional, visibility-dependent rendered ranges; these are not range-space SNR or identifiability estimates.',
            labels_read=False,new_physics_or_network_training=False,per_object_statistics_LOCAL=True,elapsed_s=time.monotonic()-start)
        write(output/'summary.json',summary);status.update(terminal=True,completed=True,elapsed_s=time.monotonic()-start);write(output/'execution.json',status);return record(output/'summary.json')
    except BaseException as exc:
        write(output/'per-object-statistics-LOCAL-PARTIAL.json',objects);write(output/'failure.json',dict(completed=False,error=type(exc).__name__,message=str(exc)))
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-start);write(output/'execution.json',status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for key in ['output','output-prefix','execution','execution-sha256','primary','primary-sha256']:p.add_argument('--'+key,required=True)
    p=sub.add_parser('run');p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='prepare':value=prepare(a.output,dict(path=a.execution,sha256=a.execution_sha256),dict(path=a.primary,sha256=a.primary_sha256),a.output_prefix)
    else:value=run(a.protocol,a.protocol_sha256)
    print(json.dumps(value))
