"""Frozen, bounded matched static linear/nonlinear qualification. No inference."""
from __future__ import annotations
import argparse,ast,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import eigsh,ArpackNoConvergence
from experiments import campaign_surface_p2_static_v01 as linear

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/clamped-surface-p2-qualification-v01'
PUBLIC=ROOT/'reports/clamped-surface-p2-qualification-v01'
CASES=[dict(index=i,E_Pa=E,K_N_m=K) for i,(E,K) in enumerate([(2000.,5.),(2000.,30.),(8000.,5.),(8000.,30.),(4000.,float(np.sqrt(150.)))])]
RESOURCES=dict(cpu_affinity=[4,5,6,7],threads=1,GPU=False,wall_cap_s=1800,RSS_cap_bytes=4*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=2*1024**3)
GATES=dict(minimum_J=.8,maximum_absolute_symmetric_strain_eigenvalue=.05,minimum_nonrigid_RMS_m=100e-6,minimum_nonrigid_fraction=.3,spatial_order_floor=1.,maximum_conditional_fine_indicator_m=20e-6,maximum_medium_indicator_m=20e-6,maximum_linear_vs_nonlinear_RMS_m=20e-6,equilibrium_absolute_N=1e-10,equilibrium_relative=1e-8,SPD_eigsh_tolerance=1e-6,SPD_eigsh_maxiter=10000,SPD_initial_seed=2026102411,maximum_NH24_vs_NH80_RMS_m=2e-6,maximum_surface_operator_relative_error=1e-10)


class ResourceStop(BaseException):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sources():
    """Pin every recursively imported repository Python source, including tests."""
    pending=['experiments/run_clamped_surface_p2_qualification_v01.py','tests/test_clamped_surface_p2_qualification_v01.py','tests/test_surface_p2_static_v01.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('Missing source '+name)
        seen.add(name);tree=ast.parse(path.read_text());package=name.rsplit('/',1)[0].replace('/','.')
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            if isinstance(node,ast.ImportFrom):
                parent=package.split('.')
                prefix='.'.join(parent[:len(parent)-node.level+1]) if node.level else ''
                base='.'.join(x for x in (prefix,node.module or '') if x)
                modules=[base]+[base+'.'+x.name for x in node.names]
            for mod in modules:
                rel=mod.replace('.','/')+'.py'
                if (ROOT/rel).is_file() and rel not in seen:pending.append(rel)
    return {p:sha(ROOT/p) for p in sorted(seen)}


def guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or not set(os.sched_getaffinity(0))<=set(RESOURCES['cpu_affinity']):raise ValueError('CPU4..7/one-thread/noGPU guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['RSS_cap_bytes']:raise ResourceStop('RSS cap exceeded')
    import shutil
    if shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise ResourceStop('Disk reserve failed')
    if LOCAL.exists() and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise ResourceStop('Output budget exceeded')


def predecessor():
    base=ROOT/'runs/operator-dev/clamped-surface-qualification-v01'
    old=json.loads((base/'summary.json').read_text());execution=json.loads((base/'execution.json').read_text())
    if not old['completed'] or not old['all10_retained'] or old['qualified_cases']!=0 or execution['status']!='COMPLETED' or not execution['completed']:
        raise ValueError('Complete retained T4 negative predecessor required')
    if sha(base/'protocol.json')!='3e3b841e7f9081b81e42690067c8f7f28b48f94d1a8dcb381a0705beecdd2767':raise ValueError('T4 predecessor protocol mismatch')
    try:os.kill(execution['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('T4 owned PID still live')
    paths=[base/name for name in ('protocol.json','summary.json','execution.json','publication-inventory-completed.json')]
    return {str(path.relative_to(ROOT)):sha(path)for path in paths}


def prepare():
    guard()
    if (LOCAL/'protocol.json').exists():raise ValueError('Frozen namespace already exists')
    p=dict(schema='clamped-surface-p2-qualification-v01',predecessor_sha256=predecessor(),cases=CASES,loads=['normal','paired_axial'],levels=[1,2,4],resources=RESOURCES,gates=GATES,linear_design=linear.design(),nonlinear_design=linear.design(),source_sha256=sources(),expected_paired_solves=30,expected_NH80_reference_solves=30,quadrature_orders=[24,80],surface_quadrature_production=16,surface_quadrature_reference=20,
           metric='Coordinate RMS sqrt(integral||difference||²/(3V)); common finest positiveKeast24; proper weightedSO3 fit',
           spatial='p=log2(dCM/dMF); fine_indicator=dMF/(2**p-1), only if p>=1; medium_indicator=2*dMF. Conditional extrapolation, not rigorous error bound.',
           all_cases_retained=True,failures_stop_inference=True,no_retuning=True,no_inference=True)
    write(LOCAL/'protocol.json',p)
    return dict(protocol=str(LOCAL/'protocol.json'),sha256=sha(LOCAL/'protocol.json'))


def weighted_rigid_metrics(points,displacement,weights):
    x=np.asarray(points,float);u=np.asarray(displacement,float);w=np.asarray(weights,float)
    if x.shape!=u.shape or x.ndim!=2 or x.shape[1]!=3 or w.shape!=(len(x),) or np.any(w<=0) or not all(np.isfinite(z).all() for z in (x,u,w)):raise ValueError('Finite positive weighted3D fields')
    w=w/w.sum();y=x+u;cx=w@x;cy=w@y;X=x-cx;Y=y-cy
    a,_,bt=np.linalg.svd(X.T@(w[:,None]*Y));fix=np.eye(3);fix[-1,-1]=np.linalg.det(a@bt);rotation=a@fix@bt
    residual=Y-X@rotation;total=np.sqrt(np.sum(w[:,None]*u*u)/3);nonrigid=np.sqrt(np.sum(w[:,None]*residual*residual)/3)
    return dict(total_RMS_m=float(total),nonrigid_RMS_m=float(nonrigid),nonrigid_fraction=float(nonrigid/total) if total>0 else 0.,proper_rotation_determinant=float(np.linalg.det(rotation)))


def spatial_gate(dCM,dMF):
    if not np.isfinite([dCM,dMF]).all() or min(dCM,dMF)<0:raise ValueError('Finite nonnegative spatial distances')
    p=float(np.log2(dCM/dMF)) if dCM>0 and dMF>0 else None
    fine=dMF/(2**p-1) if p is not None and p>=GATES['spatial_order_floor'] else None
    medium=2*dMF
    return dict(coarse_medium_RMS_m=dCM,medium_fine_RMS_m=dMF,observed_order=p,conditional_fine_indicator_m=fine,medium_indicator_m=medium,
                spatial_passed=bool(fine is not None and fine<=GATES['maximum_conditional_fine_indicator_m']),medium_adequate=bool(p is not None and p>=GATES['spatial_order_floor'] and medium<=GATES['maximum_medium_indicator_m']),conditional_only=True)


def common_rms(fields):
    """Each tuple (meshlevel,nodalfield); evaluate every pair on same finest rule."""
    sums=np.zeros((len(fields),len(fields)));volume=0.
    for points,w in linear.bulk_query_blocks(4):
        values=[linear.field_queries(n,u,points)for n,u in fields];volume+=w.sum()
        for i in range(len(fields)):
            for j in range(i):sums[i,j]+=np.sum(w[:,None]*(values[i]-values[j])**2)
    return np.sqrt(sums/(3*volume))


def deformation(forward,u):
    # Rigid fit uses exact second moments of P1 fields via positive Keast24.
    points=[];weights=[];values=[]
    for q,w in linear.bulk_query_blocks(4):points.append(q);weights.append(w);values.append(linear.field_queries(forward.n,u,q))
    result=weighted_rigid_metrics(np.concatenate(points),np.concatenate(values),np.concatenate(weights))
    samples=linear.strain_metrics(forward,u)
    result.update({key:float(np.asarray(samples[key]).reshape(-1)[0])for key in ('minimum_J','maximum_absolute_symmetric_strain_eigenvalue')})
    result.update(checked_nodes_and24and80=True,whole_element_J_certified=False)
    result['physical_passed']=bool(result['minimum_J']>=GATES['minimum_J'] and result['maximum_absolute_symmetric_strain_eigenvalue']<=GATES['maximum_absolute_symmetric_strain_eigenvalue'] and result['nonrigid_RMS_m']>=GATES['minimum_nonrigid_RMS_m'] and result['nonrigid_fraction']>=GATES['minimum_nonrigid_fraction'])
    return result


def reaction_diagnostics(forward,u,force,E,K,nonlinear_residual=None):
    residual=((E*forward.Kunit+K*forward.W)@u.ravel()).reshape(u.shape)-force if nonlinear_residual is None else nonlinear_residual
    support=-(K*forward.W@u.ravel()).reshape(u.shape);clamp=np.zeros_like(u);clamp[forward.fixed]=residual[forward.fixed]
    external=force+support+clamp;x=forward.X+u if nonlinear_residual is not None else forward.X
    tolerance=max(GATES['equilibrium_absolute_N'],GATES['equilibrium_relative']*np.linalg.norm(force))
    balance_tolerance=np.sqrt(len(u))*tolerance+1e-12*max(np.linalg.norm(force),1.)
    force_error=float(np.linalg.norm(external.sum(0)));moment_error=float(np.linalg.norm(np.cross(x,external).sum(0)));moment_tolerance=float(np.max(np.linalg.norm(x,axis=1))*balance_tolerance)
    norm=float(np.linalg.norm(residual.ravel()[forward.free]))
    return dict(free_residual_N=norm,tolerance_N=tolerance,force_balance_error_N=force_error,force_balance_tolerance_N=balance_tolerance,moment_balance_error_Nm=moment_error,moment_balance_tolerance_Nm=moment_tolerance,exact_clamp=bool(np.all(u[forward.fixed]==0)),passed=bool(norm<=tolerance and force_error<=balance_tolerance and moment_error<=moment_tolerance and np.all(u[forward.fixed]==0)))


def quadrature_checks(forward):
    rows={};matrices={}
    for order in (4,8,12,16,20):
        matrices[order],rows[str(order)]=linear.consistent_support(forward.X,forward.faces,order=order)
    norm=lambda A:float(__import__('scipy').sparse.linalg.norm(A))
    e812=norm(matrices[8]-matrices[12])/norm(matrices[12]);e1216=norm(matrices[12]-matrices[16])/norm(matrices[16]);e1620=norm(matrices[16]-matrices[20])/norm(matrices[20])
    loads=linear.qualification_loads(forward.n);load_rows={}
    for name,f in loads.items():
        expected=np.array([0,0,-.1])if name=='normal'else np.array([.48,0,0]);force_error=float(np.linalg.norm(f.sum(0)-expected));moment=float(np.linalg.norm(np.cross(forward.X-[.18,.04,.04],f).sum(0)))
        load_rows[name]=dict(net_force_error_N=force_error,reference_moment_error_Nm=moment,passed=bool(force_error<=linear.shared.GATES['load_net_force_error_N'] and moment<=linear.shared.GATES['load_moment_error_Nm']))
    passed=rows['16']['normalization_relative_error']<=linear.shared.GATES['support_normalization_relative_error'] and e1620<=GATES['maximum_surface_operator_relative_error'] and all(x['passed']for x in load_rows.values())
    return dict(passed=bool(passed),production_order=16,independent_order=20,orders=rows,relative_Frobenius_8_12=e812,relative_Frobenius_12_16=e1216,old12_vs16_control_passed=bool(e1216<=GATES['maximum_surface_operator_relative_error']),relative_Frobenius_16_20=e1620,loads=load_rows,preproduction_quadrature_refinement_only=True)


def stability_diagnostic(reduced,n):
    eigenvalues,eigenvectors=eigsh(reduced,k=1,which='SA',return_eigenvectors=True,tol=GATES['SPD_eigsh_tolerance'],maxiter=GATES['SPD_eigsh_maxiter'],v0=np.random.default_rng(GATES['SPD_initial_seed']+n).normal(size=reduced.shape[0]))
    eigenvalue=float(eigenvalues[0]);vector=eigenvectors[:,0]
    eigen_residual=float(np.linalg.norm(reduced@vector-eigenvalue*vector))
    relative_eigen_residual=eigen_residual/max(abs(eigenvalue),np.finfo(float).tiny)
    return dict(completed=True,minimum_eigenvalue_estimate_N_m=eigenvalue,eigenpair_residual_N_m=eigen_residual,relative_eigenpair_residual=relative_eigen_residual,positive_Ritz_lower_endpoint_N_m=eigenvalue-eigen_residual,
                passed=bool(eigenvalue-eigen_residual>0 and relative_eigen_residual<=GATES['SPD_eigsh_tolerance']),scope='Converged smallest-algebraic eigsh diagnostic; no interval arithmetic proof')


def run(digest):
    guard();p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_sha256']!=sources() or p['predecessor_sha256']!=predecessor():raise ValueError('Frozen protocol/source/predecessor mismatch')
    if (LOCAL/'execution.json').exists():raise ValueError('No retries in frozen namespace')
    def timeout(*_):raise ResourceStop('Frozen global wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_cap_s']);start=time.monotonic()
    write(LOCAL/'execution.json',dict(status='RUNNING',pid=os.getpid(),protocol_sha256=digest))
    per_case=[];forwards={n:linear.SparseStaticForward(n)for n in p['levels']};quads={str(n):quadrature_checks(f)for n,f in forwards.items()}
    for case in CASES:
        for load_name in p['loads']:
            record=dict(case_index=case['index'],load=load_name,levels={},completed=True);fields={};E=case['E_Pa'];K=case['K_N_m']
            for n in p['levels']:
                guard();fwd=forwards[n];force=linear.qualification_loads(n)[load_name];row={};fields[n]={}
                try:
                    u=fwd.staticforward(np.array([[E,K]]),force)[0,0];fields[n]['linear']=u
                    row['linear']=dict(deformation=deformation(fwd,u),equilibrium=reaction_diagnostics(fwd,u,force,E,K))
                    model=linear.P2NonlinearReference.from_forward(fwd,E,K,quadrature_order=24)
                    un,diagnostics=model.solve(force);fields[n]['nonlinear']=un
                    state=model.evaluate(un,force,tangent=True);residual=state['residual'];H=state['tangent']
                    try:
                        stability=stability_diagnostic(H[model.free][:,model.free],n)
                    except Exception as exc:stability=dict(completed=False,passed=False,error=type(exc).__name__+': '+str(exc))
                    model80=linear.P2NonlinearReference.from_forward(fwd,E,K,quadrature_order=80)
                    un80,diag80=model80.solve(force)
                    row['NH24_vs_NH80_RMS_m']=float(common_rms([(n,un),(n,un80)])[1,0])
                    row['NH24_vs_NH80_passed']=row['NH24_vs_NH80_RMS_m']<=GATES['maximum_NH24_vs_NH80_RMS_m']
                    state80=model80.evaluate(un80,force,tangent=True)
                    try:stability80=stability_diagnostic(state80['tangent'][model80.free][:,model80.free],n)
                    except Exception as exc:stability80=dict(completed=False,passed=False,error=type(exc).__name__+': '+str(exc))
                    row['NH80_reference']=dict(Newton=diag80,deformation=deformation(fwd,un80),equilibrium=reaction_diagnostics(fwd,un80,force,E,K,state80['residual']),stability=stability80)
                    row['nonlinear']=dict(deformation=deformation(fwd,un),equilibrium=reaction_diagnostics(fwd,un,force,E,K,residual),Newton=diagnostics,stability=stability)
                    row['linear_vs_nonlinear_RMS_m']=float(common_rms([(n,u),(n,un)])[1,0]);row['linear_vs_nonlinear_passed']=row['linear_vs_nonlinear_RMS_m']<=GATES['maximum_linear_vs_nonlinear_RMS_m']
                    np.savez_compressed(LOCAL/f'case-{case["index"]:02d}-{load_name}-n{n}-LOCAL.npz',linear=u,nonlinear=un,nonlinear80=un80)
                except Exception as exc:
                    row['completed']=False;row['error']=type(exc).__name__+': '+str(exc);record['completed']=False
                record['levels'][str(n)]=row;guard()
            record['spatial']={}
            for law in ('linear','nonlinear'):
                if all(law in fields[n]for n in p['levels']):
                    dist=common_rms([(n,fields[n][law])for n in p['levels']]);record['spatial'][law]=spatial_gate(float(dist[1,0]),float(dist[2,1]))
                else:record['spatial'][law]=dict(spatial_passed=False,medium_adequate=False,error='Incomplete retained fields')
            record['passed']=bool(record['completed'] and all(quads[str(n)]['passed']for n in p['levels']) and all(all(row[law]['deformation']['physical_passed'] and row[law]['equilibrium']['passed']for law in ('linear','nonlinear')) and row['nonlinear']['stability']['passed'] and row['linear_vs_nonlinear_passed']and row['NH24_vs_NH80_passed']and row['NH80_reference']['deformation']['physical_passed']and row['NH80_reference']['equilibrium']['passed']and row['NH80_reference']['stability']['passed']for row in record['levels'].values()) and all(x['spatial_passed']and x['medium_adequate']for x in record['spatial'].values()))
            per_case.append(record);write(LOCAL/f'case-{case["index"]:02d}-{load_name}-summary.json',record)
    summary=dict(completed=True,paired_cases=10,paired_mesh_solves=30,NH80_reference_solves=30,all10_retained=len(per_case)==10,qualified_cases=sum(x['passed']for x in per_case),all_qualified=all(x['passed']for x in per_case),protocol_sha256=digest,quadrature=quads,per_case=per_case,wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,no_inference_run=True,no_continuum_or_dynamic_claim=True,raw_arrays_LOCAL=True)
    write(LOCAL/'summary.json',summary);write(PUBLIC/'summary.json',summary)
    write(LOCAL/'execution.json',dict(status='COMPLETED',pid=os.getpid(),protocol_sha256=digest,completed=True,qualified=summary['all_qualified'],wall_s=summary['wall_s']))
    signal.alarm(0);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--protocol-sha256');a=parser.parse_args()
    try:
        result=prepare()if a.mode=='prepare'else run(a.protocol_sha256)
    except BaseException as exc:
        if a.mode=='run':write(LOCAL/'execution.json',dict(status='FAILED',completed=False,error=type(exc).__name__+': '+str(exc),pid=os.getpid()))
        raise
    print(json.dumps({k:v for k,v in result.items()if k not in ('per_case','quadrature')}))
