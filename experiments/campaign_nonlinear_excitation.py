"""Prospective finite-strain excitation development; no cohort or learned model.

Immutable nonlinear reference reused; CPU2 only. All failed design amplitudes
and parameter cases retained. Gaussian controls are material-frame distributed
loads, identical across hypotheses and conforming meshes, not pure torques.
"""
from __future__ import annotations
import argparse, hashlib, json, signal, time
from pathlib import Path
import numpy as np
import torch
try:
    from .campaign_nonlinear_gaussian_reference import NonlinearGaussianReference, tensor, StepFailure
    from .campaign_compliant_gaussian_linear import tiny_geometry,candidate_patches
    from . import campaign_boundary_gaussians as boundary
except ImportError:
    from campaign_nonlinear_gaussian_reference import NonlinearGaussianReference,tensor,StepFailure
    from campaign_compliant_gaussian_linear import tiny_geometry,candidate_patches
    import campaign_boundary_gaussians as boundary

PATTERNS=('legacy_single','balanced_axial','transverse_couple_1','transverse_couple_2')
SOURCE=Path(__file__)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def boundary_faces(tets):
    counts={}
    for tet in tets:
        for j in range(4):
            f=tuple(sorted(np.delete(tet,j)));counts[f]=counts.get(f,0)+1
    return np.asarray([f for f,n in sorted(counts.items()) if n==1],dtype=np.int64)

def refine(points,tets):
    """Conforming1-to8 split; shared global edge midpoints, no solved-state repair."""
    vertices=list(np.asarray(points).copy()); edges={};children=[]
    def mid(a,b):
        edge=tuple(sorted((int(a),int(b))))
        if edge not in edges:edges[edge]=len(vertices);vertices.append((points[a]+points[b])/2)
        return edges[edge]
    for a,b,c,d in tets:
        ab,ac,ad,bc,bd,cd=[mid(i,j) for i,j in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]]
        children += [[a,ab,ac,ad],[ab,b,bc,bd],[ac,bc,c,cd],[ad,bd,cd,d],
                     [ab,ac,ad,cd],[ab,ac,bc,cd],[ab,ad,bd,cd],[ab,bc,bd,cd]]
    X=np.asarray(vertices);T=np.asarray(children,dtype=np.int64)
    for row in T:
        if np.linalg.det((X[row[1:]]-X[row[0]]).T)<0:row[1],row[2]=row[2],row[1]
    return X,T,boundary_faces(T)

def geometry(level=0):
    X,T,F=tiny_geometry()
    for _ in range(level):X,T,F=refine(X,T)
    return X,T,F

def build_model(young=5000.,fixture_scale=2.,mesh_level=0):
    X,T,F=geometry(mesh_level);patches,K,D=candidate_patches()
    loads=[boundary.GaussianPatch(p.center_material_m,p.covariance_material_m2,'known_hardware') for p in patches]
    return NonlinearGaussianReference(X,T,F,young=young,poisson=.35,density=625.,support_patches=patches,
        stiffness=K*fixture_scale,damping=D,load_patches=loads,minimum_J=.1)

def command(pattern,t,amplitude,duration=.2):
    """Fixed world force function, independent of E/K, mesh and observations."""
    patches,_,_=candidate_patches();axis=patches[1].center_material_m-patches[0].center_material_m
    axis=axis/np.linalg.norm(axis);helper=np.eye(3)[np.argmin(abs(axis))]
    transverse=np.cross(axis,helper);transverse/=np.linalg.norm(transverse)
    third=np.cross(axis,transverse)
    axes={'legacy_single':transverse,'balanced_axial':axis,'transverse_couple_1':transverse,'transverse_couple_2':third}
    if pattern not in axes or not np.isfinite(amplitude) or amplitude<0:raise ValueError('Declared nonnegative known command required')
    s=np.sin(np.pi*t/duration)**2 if 0<=t<=duration else 0.
    first=amplitude*s*axes[pattern]
    return np.stack([first,np.zeros(3) if pattern=='legacy_single' else -first])

def nonrigid_rms(rest,x):
    a=rest-rest.mean(0);b=x-x.mean(0);U,_,V=np.linalg.svd(a.T@b)
    fix=np.eye(3);fix[2,2]=np.linalg.det(U@V)
    return float(np.sqrt(np.mean((b-a@(U@fix@V))**2)))

def tangent_step(model,xnew,sx,sv,h):
    """Original implicit stationarity tangent, logE and log fixture stiffness.

    M,C independent of these parameters. Kfixture depends linearly on scale;
    U_NH depends linearly on E. This is not a posterior or a fitted Jacobian.
    """
    a=xnew.detach().requires_grad_(True)
    grad=torch.autograd.grad(model.elastic(a)[0],a)[0].flatten().detach()
    H=torch.autograd.functional.hessian(lambda z:model.elastic(z.reshape_as(a))[0],a.flatten()).detach()
    M=torch.diag(model.mass);C=model.fixture.C
    H=H+M/h**2+C/h+model.fixture.K
    direct=torch.stack([grad,model.fixture.K@(xnew-model.rest).flatten()],dim=1)
    new_s=torch.linalg.solve(H,(M/h**2+C/h)@sx+M@sv/h-direct)
    return new_s,(new_s-sx)/h

def simulate(young,fixture_scale,pattern,amplitude,hz=120,mesh_level=0,*,sensitivities=False,duration=.2):
    model=build_model(young,fixture_scale,mesh_level);h=1/hz
    x=model.rest.clone();v=torch.zeros_like(x);targets=x[None].repeat(2,1,1)
    states=[x.numpy().copy()];velocities=[v.numpy().copy()];forces=[];held_commands=[];moments=[];resultants=[];ledgers=[]
    sx=torch.zeros((x.numel(),2),dtype=torch.float64);sv=torch.zeros_like(sx);jac=[sx.numpy().reshape(*x.shape,2)]
    failure=None
    for k in range(round(duration*hz)):
        held=command(pattern,(k+.5)*h,amplitude,duration);f=model.known_force(held)
        start=x.clone()
        try:x,v,row=model.step(x,v,targets,targets,f,h)
        except TimeoutError:raise
        except Exception as exc:
            failure={'step_index':k+1,'time_s':(k+1)*h,'exception':repr(exc),'history':getattr(exc,'history',[])};break
        if row['force_residual_N']>row['threshold_N']*1.0001:
            failure={'step_index':k+1,'exception':'Original physical residual exceeded declared acceptance','ledger':row};break
        if sensitivities:
            sx,sv=tangent_step(model,x,sx,sv,h);jac.append(sx.numpy().reshape(*x.shape,2))
        states.append(x.numpy().copy());velocities.append(v.numpy().copy());forces.append(f.numpy())
        held_commands.append(held);resultants.append(f.sum(0).numpy());moments.append(torch.stack([torch.linalg.cross(start,f).sum(0),torch.linalg.cross(x,f).sum(0)]).numpy())
        ledgers.append({k:v for k,v in row.items() if k!='history'})
    q=np.asarray(states);transport=model.kernels.transport(tensor(q));sing=np.linalg.svd(transport.deformation_gradients.numpy(),compute_uv=False)-1
    nonrigid=[nonrigid_rms(model.rest.numpy(),a) for a in q]
    summary={'completed':failure is None,'young_Pa':young,'fixture_scale':fixture_scale,'pattern':pattern,'amplitude_N_per_patch':amplitude,'hz':hz,'mesh_level':mesh_level,'tetrahedra':len(model.tets),'steps_completed':len(ledgers),
      'maximum_absolute_principal_stretch_strain':float(abs(sing).max()),'minimum_principal_stretch_strain':float(sing.min()),'maximum_principal_stretch_strain':float(sing.max()),
      'peak_nonrigid_coordinate_RMS_m':max(nonrigid),'peak_total_coordinate_displacement_RMS_m':float(np.sqrt(np.mean((q-model.rest.numpy())**2,axis=(1,2))).max()),
      'minimum_J':float(transport.jacobians.min()),'maximum_original_residual_N':max((r['force_residual_N'] for r in ledgers),default=0.),
      'maximum_balance_minus_residual_work_J':max((abs(r['balance_error_J']-r['free_residual_work_J']) for r in ledgers),default=0.),
      'maximum_resultant_N':float(np.linalg.norm(resultants,axis=-1).max()) if resultants else 0.,'maximum_actual_world_moment_Nm':float(np.linalg.norm(moments,axis=-1).max()) if moments else 0.,
      'maximum_gaussian_mass_error_kg':float((transport.bulk_density*transport.volumes-model.kernels.masses).abs().max()),'failure':failure}
    summary['excitation_target_passed']=bool(summary['completed'] and .02<=summary['maximum_absolute_principal_stretch_strain']<=.08 and summary['peak_nonrigid_coordinate_RMS_m']>=.001)
    arrays={'times_s':np.arange(len(q))/hz,'positions_m':q,'velocities_m_s':np.asarray(velocities),'known_nodal_forces_N':np.asarray(forces),'held_patch_forces_N':np.asarray(held_commands),'held_force_midpoints_s':(np.arange(len(held_commands))+.5)/hz,'actual_resultants_N':np.asarray(resultants),'actual_start_end_moments_Nm':np.asarray(moments),'rest_m':model.rest.numpy(),'tets':model.tets.numpy()}
    if sensitivities:arrays['position_logparameter_jacobian_m']=np.asarray(jac)
    return summary,arrays,ledgers

def column_metrics(J):
    J=np.asarray(J).reshape(-1,2);n=np.linalg.norm(J,axis=0);s=np.linalg.svd(J,compute_uv=False)
    cosine=float(J[:,0]@J[:,1]/np.prod(n)) if np.all(n>0) else None
    ratio=float(s[1]/s[0]) if s[0]>0 else 0.
    return {'column_norms_m':n.tolist(),'signed_cosine':cosine,'absolute_cosine':abs(cosine) if cosine is not None else None,'singular_values_m':s.tolist(),'small_to_large_singular_value_ratio':ratio,'separation_diagnostic_passed':cosine is not None and abs(cosine)<=.9 and ratio>=.1}

def affine_patch_check():
    """Operating-strain analytic P and conforming mesh affine-energy invariance."""
    F=tensor([[1.06,.025,0],[.01,.97,.015],[0,0,1.02]]);U=[];gradient_errors=[]
    for level in (0,1,2):
        m=build_model(mesh_level=level);x=(m.rest@F.T+tensor([.02,-.01,.01])).requires_grad_(True)
        energy,J=m.elastic(x);grad=torch.autograd.grad(energy,x)[0]
        mu=m.young/(2*(1+m.poisson));lam=m.young*m.poisson/((1+m.poisson)*(1-2*m.poisson));invT=torch.linalg.inv(F).T
        P=mu*(F-invT)+lam*torch.log(torch.linalg.det(F))*invT;expected=torch.zeros_like(x)
        for e,ids in enumerate(m.tets):expected[ids]+=m.volumes[e]*(m.gradients[e]@P.T)
        gradient_errors.append(float((grad-expected).abs().max()));U.append(float(energy.detach()))
    return {'energies_J':U,'maximum_energy_mesh_spread_J':max(U)-min(U),'maximum_analytic_P_gradient_error_N':max(gradient_errors),'passed':max(U)-min(U)<1e-12 and max(gradient_errors)<1e-10,'scope':'Affine patch identity, not dynamic mesh convergence'}

def prepare(path):
    deps=[SOURCE,Path('experiments/campaign_nonlinear_gaussian_reference.py'),Path('experiments/campaign_nonlinear_fem.py'),Path('experiments/campaign_linear_fem.py'),Path('experiments/campaign_boundary_gaussians.py'),Path('experiments/campaign_compliant_gaussian_linear.py'),Path('experiments/campaign_material_gaussians.py')]
    d={'schema':'prospective-nonlinear-excitation-development-v0.1','status':'FROZEN_BEFORE_DESIGN_PILOTS','CPU_threads':2,'wall_cap_s':600,'duration_s':.2,
       'pilot':{'young_Pa':5000.,'fixture_scale':2.,'hz':120,'patterns':list(PATTERNS),'amplitudes_N_per_patch':[.25,1.,4.],'budget_cases':12,'selection':'Smallest amplitude meeting ALL2–8% absolute principal-stretch strain,1mm best-rigid-removed coordinateRMS,completedpositiveJ; else no accepted pattern'},
       'factorial':{'young_Pa':[2500.,5000.,10000.],'fixture_scale':[2.,10.],'hz':120,'cases_per_accepted_pattern':6,'known_commands_identical_across_hypotheses':True,'all_cases_retained_even_target_failures':True},
       'temporal':{'nominal_only':True,'rates_hz':[60,120,240],'full_trajectory_error_tolerance':'max(2e-6m,.05*finest peak nonrigidcoordinateRMS)','require_final_adjacent_contraction':True},
       'mesh':{'pattern_priority':['balanced_axial','transverse_couple_1','transverse_couple_2','legacy_single'],'nominal_young_Pa':5000.,'nominal_fixture_scale':2.,'hz':240,'initial_tet_counts':[2,16],'if_initial_fails_one_continuation_to_tets':128,'held_original_nodes':[1,4],'tolerance':'max(2e-6m,.1*fine heldnode displacementcoordinateRMS)','same_physical_patch_centers_covariances_and_force_function':True,'coarse_failure_never_relabelled':True},
       'sensitivity':{'parameters':['logYoung','logFixtureScale'],'method':'Exact implicit-stationarity tangent, no extra physical runs','sparse_nodes':[1,4],'sparse_times_s':[.05,.1,.15,.2],'nominal_young_Pa':5000.,'nominal_fixture_scale':2.,'absolute_column_cosine_limit':.9,'singular_value_ratio_minimum':.1,'not_inference_or_identifiability_guarantee':True},
       'common_parameters':{'poisson':.35,'density_kg_m3':625.,'damping':'Fixed candidate_patches D, not scaled with fixture stiffness','Gaussian_patches':'Immutable candidate_patches material centers/covariances; vertexlumped restsurface quadrature','gravity_m_s2':[0,0,0],'contact':'Unsupported; no interaction with plane/contact primitive','minimum_J':.1,'force_law':'F(t)=A sin²(pi*t/.2) times fixedworldunitdirection, oppositepair exceptlegacy; heldmidpoint approximation for eachsolverinterval','force_axes':'Derived ONLY from fixedphysicalpatch centers, independentofmesh andparameters','rotation_note':'Transversecouplesarefixedworldforcepairs; actualworldmoment isreported perendpoint and neednotbeconstant'},
       'source_sha256':{str(p):sha(p) for p in deps},'limits':['Tiny discrete nonlinear development, no current cohort reads','Mesh/temporalfailuresblockcorrespondingforwardqualification','No inverse, neuraltraining, nativeIsaac orcontinuumclaim']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(d,indent=2)+'\n');return d

def run(protocol_path,protocol_sha,output):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Changed protocol')
    protocol=json.loads(Path(protocol_path).read_text())
    for p,digest in protocol['source_sha256'].items():
        if sha(p)!=digest:raise ValueError('Changed source '+p)
    torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();summary={'completed':False,'protocol':{'path':str(protocol_path),'sha256':protocol_sha},'pilots':[],'factorial':[],'temporal':{},'sensitivity':{},'mesh':{},'all_failures_retained':True}
    def save():
        summary['elapsed_s']=time.monotonic()-start;(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def alarm(sig,frame):raise TimeoutError('Prospective600sCPU2budgetexhausted; no retry')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(protocol['wall_cap_s'])
    def capture(label,E,K,p,A,hz=120,level=0,sens=False):
        result,arrays,ledger=simulate(E,K,p,A,hz,level,sensitivities=sens)
        np.savez_compressed(out/(label+'.npz'),**arrays)
        (out/(label+'.json')).write_text(json.dumps({'summary':result,'steps':ledger},indent=2)+'\n')
        result['artifact_npz']=str(out/(label+'.npz'));return result,arrays
    try:
        summary['affine_patch']=affine_patch_check();save();accepted={}
        for p in PATTERNS:
            candidates=[]
            for A in protocol['pilot']['amplitudes_N_per_patch']:
                r,a=capture(f'pilot-{p}-{A:g}',5000.,2.,p,A);summary['pilots'].append(r);candidates.append(r);save()
            accepted[p]=next((r['amplitude_N_per_patch'] for r in candidates if r['excitation_target_passed']),None)
        summary['accepted_amplitudes_N_per_patch']=accepted;save();jacobians=[]
        for p,A in accepted.items():
            if A is None:continue
            nominal=None
            for E in protocol['factorial']['young_Pa']:
                for K in protocol['factorial']['fixture_scale']:
                    r,a=capture(f'factor-{p}-E{E:g}-K{K:g}',E,K,p,A,sens=(E==5000. and K==2.));summary['factorial'].append(r);save()
                    if E==5000. and K==2.:nominal=(r,a)
            temporal=[capture(f'time-{p}-{hz}',5000.,2.,p,A,hz) for hz in (60,240)]
            levels=[temporal[0],nominal,temporal[1]]
            if all(r['completed'] for r,_ in levels):
                q=[a['positions_m'] for _,a in levels];errors=[float(np.sqrt(np.mean((q[i]-q[i+1][::2])**2))) for i in range(2)];tol=max(2e-6,.05*levels[-1][0]['peak_nonrigid_coordinate_RMS_m'])
                summary['temporal'][p]={'rates_hz':[60,120,240],'full_trajectory_adjacent_RMSE_m':errors,'tolerance_m':tol,'passed':errors[-1]<=tol and errors[-1]<errors[0],'scope':'NominalE5000/K2only; notfactorialenvelopeproof'}
            else:summary['temporal'][p]={'passed':False,'reason':'Retained failed temporal cell'}
            if nominal[0]['completed']:
                J=nominal[1]['position_logparameter_jacobian_m'];rows=np.array([6,12,18,24]);sparse=J[rows][:,[1,4]].reshape(-1,2)
                summary['sensitivity'][p]={'sparse':column_metrics(sparse),'full_trajectory':column_metrics(J)};jacobians.append(sparse)
            save()
        if jacobians:summary['sensitivity']['stacked_patterns_sparse']=column_metrics(np.concatenate(jacobians))
        chosen=next((p for p in protocol['mesh']['pattern_priority'] if accepted[p] is not None),None)
        if chosen:
            A=accepted[chosen];r0,a0=capture('mesh2-'+chosen,5000.,2.,chosen,A,240,0)
            r1,a1=capture('mesh16-'+chosen,5000.,2.,chosen,A,240,1)
            def comparison(rcoarse,acoarse,rfine,afine):
                if not (rcoarse['completed'] and rfine['completed']):return {'passed':False,'reason':'Retained failed mesh solve'}
                # Originalverticesarepreservedasfirst5 inallconformingmeshes.
                delta=acoarse['positions_m'][:,[1,4]]-afine['positions_m'][:,[1,4]]
                signal_rms=float(np.sqrt(np.mean((afine['positions_m'][:,[1,4]]-afine['rest_m'][[1,4]])**2)));err=float(np.sqrt(np.mean(delta**2)));tol=max(2e-6,.1*signal_rms)
                return {'coarse_tetrahedra':rcoarse['tetrahedra'],'fine_tetrahedra':rfine['tetrahedra'],'heldnode_full_trajectory_RMSE_m':err,'fine_held_displacement_RMS_m':signal_rms,'tolerance_m':tol,'passed':err<=tol}
            initial=comparison(r0,a0,r1,a1);summary['mesh']={'pattern':chosen,'amplitude_N_per_patch':A,'initial_2_to_16':initial,'fine_excitation':r1,'qualification':'2tetunqualifiedifinitialfails'};save()
            if not initial['passed']:
                r2,a2=capture('mesh128-'+chosen,5000.,2.,chosen,A,240,2)
                summary['mesh']['continuation_16_to_128']=comparison(r1,a1,r2,a2);summary['mesh']['finest_excitation']=r2;save()
        else:summary['mesh']={'passed':False,'reason':'No accepted finite-strain excitation; no mesh solve authorized'}
        summary['completed']=True
        summary['qualification']={'any_accepted_excitation':any(A is not None for A in accepted.values()),'all_nominal_temporal_checks_passed':bool(accepted) and bool(summary['temporal']) and all(r['passed'] for r in summary['temporal'].values()),'coarse2tet_mesh_qualified':bool(summary['mesh'].get('initial_2_to_16',{}).get('passed',False)),'inverse_or_neural_training_authorized':False,'current_cohort_used':False}
    except Exception as exc:summary['execution_failure']={'exception':repr(exc),'history':getattr(exc,'history',[])}
    finally:signal.alarm(0);save()
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    prepare_parser=sub.add_parser('prepare');prepare_parser.add_argument('--protocol',required=True)
    r=sub.add_parser('run');r.add_argument('--protocol',required=True);r.add_argument('--protocol-sha256',required=True);r.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        result=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps(result['qualification'] if result['completed'] else result.get('execution_failure'),indent=2));raise SystemExit(0 if result['completed'] else 1)
