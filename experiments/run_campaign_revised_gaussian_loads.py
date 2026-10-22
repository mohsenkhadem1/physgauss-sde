"""Prospective consistent-Gaussian central-tension screens; no inverse/network.

Two explicit pulses only. Fixture unchanged, load standard deviations x1.5;
consistent load vectors AND fixture Gram are a declared new discretization.
All previous compression/lumped failures stay immutable.
"""
from __future__ import annotations
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import torch
try:
    from .campaign_nonlinear_excitation import nonrigid_rms
    from .diagnose_nonlinear_physical_probes import interpolate,error_metrics
except ImportError:
    from campaign_nonlinear_excitation import nonrigid_rms
    from diagnose_nonlinear_physical_probes import interpolate,error_metrics


def model_api():
    try:
        from .campaign_consistent_nonlinear_gaussian import ConsistentSurfaceNonlinearGaussian,canonical_load_definition,tension_command
    except ImportError:
        from campaign_consistent_nonlinear_gaussian import ConsistentSurfaceNonlinearGaussian,canonical_load_definition,tension_command
    return ConsistentSurfaceNonlinearGaussian,canonical_load_definition,tension_command


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rec(p):return {'path':str(p),'sha256':sha(p)}

def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,default=lambda a:np.asarray(a).tolist())+'\n')

def select_screen(records):
    """Prospective rule, never chooses an ineligible amplitude or new force."""
    eligible=[r for r in records if r['completed'] and r['minimum_J']>.1 and .02<=r['maximum_absolute_principal_stretch_strain']<=.08 and r['peak_nonrigid_coordinate_RMS_m']>=.001]
    if not eligible:return None
    best=min(abs(r['maximum_absolute_principal_stretch_strain']-.05) for r in eligible)
    tied=[r for r in eligible if abs(abs(r['maximum_absolute_principal_stretch_strain']-.05)-best)<=64*np.finfo(float).eps]
    return min(r['duration_s'] for r in tied)


def make_model(level):
    cls,_,_=model_api()
    return cls.from_design(mesh_level=level,young=5000.,poisson=.35,density=625.,fixture_scale=2.)


def prepare(path):
    _,definition,tension=model_api()
    sources=['experiments/run_campaign_revised_gaussian_loads.py','experiments/campaign_consistent_nonlinear_gaussian.py','experiments/campaign_consistent_surface_gaussians.py','experiments/diagnose_nonlinear_surface_quadrature.py','experiments/campaign_quality_tet_refinement.py','experiments/campaign_sparse_nonlinear_gaussian.py','experiments/campaign_nonlinear_excitation.py','experiments/campaign_compliant_gaussian_linear.py','experiments/diagnose_nonlinear_physical_probes.py','experiments/campaign_boundary_gaussians.py']
    d={'schema':'prospective-consistent-central-tension-v0.1','status':'FROZEN_BEFORE_ANY_SCREEN_DYNAMICS','CPU_threads':2,
       'physical_parameters':{'young_Pa':5000.,'poisson':.35,'density_kg_m3':625.,'fixture_scale':2.,'fixture_geometry_widths_K_D':'UNCHANGED','load_std_scale':1.5,'gravity':[0,0,0],'contact':'Unsupported','mass':'Originallumpedvolumemass','force_N_per_patch':1.,'worldforce':'Fixedrest-centroidalignedcentralTENSION(-axis,+axis),notfollowerforce','load_and_fixture_quadrature':'NEWconsistentpositiveGaussiansurfaceP1load/Gram'},
       'model_readiness':rec('runs/operator-dev/consistent-nonlinear-gaussian-v01/readiness.json'),'canonical_load_definition':definition(),'peak_world_force_pairs_N':tension(.2,1.,.4).tolist(),
       'screen':{'tetrahedra':128,'mesh_level':2,'hz':240,'durations_s':[.4,.8],'hard_wall_cap_s':300,'minimum_J':.1,'selection':'Completeand2–8%peakabsoluteprincipalstretchstrainand>=1mmnonrigidRMS;closest5%strain,tie64epsstrainwindowthenshorterT;otherwiseNONEandSTOP'},
       'fine':{'tetrahedra':[1024,8192],'mesh_levels':[3,4],'hz':240,'hard_wall_cap_s':600,'reuse_selected128_screen':True,'admission':'Onlyeligiblepredeclaredscreenandrecordedcostforecast<=600s','costforecast':'2xpriorold8192perstep *selectedstepcount +2xprior1024cost +30ssetup;screenmeasuredcostalsoreported'},
       'geometry_only_prior_runtime_evidence':rec('runs/gaussian-campaign-v01/nonlinear-resolution-v02/summary.json'),
       'old_gate_preserved':{'nodes':[1,4],'full_trajectory_tolerance':'max2um,10%finehelddisplacementRMS','original_noise_query_times_s':[.05,.1,.15,.2],'additional_drive_fraction_times':[.25,.5,.75,1.],'noise_threshold_maxcoordinate_m':[2e-6,20e-6],'depthscale':'Coordinateproxy,NOTrenderedranges'},
       'fixed_probe_protocol':rec('runs/operator-dev/nonlinear-physical-probes-v01/protocol.json'),'source_sha256':{p:sha(p) for p in sources},
       'limits':['No amplitudesearchorretry','Allfailedscreensretained','Noautomaticfineifneithereligible','Onlyspatialchecks;timequalificationNOTRUNinthisstage','Noinverseorneuraltrainingevenifspatialgatepasses','Alloldcompression/meshandnoisegatesremainfailedhistoricalevidence']}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    dump(p,d);return d


def validate(protocol,digest):
    if sha(protocol)!=digest:raise ValueError('Changedprospectivedesign')
    p=json.loads(Path(protocol).read_text())
    for path,h in p['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Changedsource '+path)
    for r in [p['fixed_probe_protocol'],p['geometry_only_prior_runtime_evidence'],p['model_readiness']]:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changedindependentevidence')
    return p


def capture(level,duration,path):
    if path.suffix:raise ValueError('Artifactbaserequiresnosuffix;neveroverwritefractionaldurationcaptures')
    start=time.monotonic();model=make_model(level);_,_,tension=model_api();h=1/240
    x=model.rest.copy();v=np.zeros_like(x);target=np.repeat(x[None],2,axis=0);qs=[x.copy()];vs=[v.copy()];held=[];resultants=[];moments=[];rows=[]
    maximum_strain=0.;maximum_nonrigid=0.;minimum_J=1.;failure=None;timed_out=None
    try:
        for k in range(round(duration*240)):
            control=tension((k+.5)*h,1.,duration);force=model.known_force(control);old=x.copy()
            x,v,row=model.step(x,v,target,target,force,h)
            if row['force_residual_N']>row['threshold_N']*1.0001:raise RuntimeError('Originalresidualnotaccepted')
            F=np.eye(3)+np.einsum('eai,eaJ->eiJ',(x-model.rest)[model.tets],model.gradients)
            maximum_strain=max(maximum_strain,float(np.max(abs(np.linalg.svd(F,compute_uv=False)-1))))
            maximum_nonrigid=max(maximum_nonrigid,nonrigid_rms(model.rest,x));minimum_J=min(minimum_J,row['minimum_J'])
            qs.append(x.copy());vs.append(v.copy());held.append(control);resultants.append(force.sum(0));moments.append(np.stack([np.cross(old,force).sum(0),np.cross(x,force).sum(0)]));rows.append(row)
    except TimeoutError as exc:
        failure={'exception':repr(exc),'step_attempted':len(rows)+1,'history':getattr(exc,'history',[])};timed_out=exc
    except Exception as exc:failure={'exception':repr(exc),'step_attempted':len(rows)+1,'history':getattr(exc,'history',[])}
    arrays={'positions_m':np.asarray(qs),'velocities_m_s':np.asarray(vs),'times_s':np.arange(len(qs))/240,'rest_m':model.rest,'tets':model.tets,'held_patch_forces_N':np.asarray(held),'actual_resultants_N':np.asarray(resultants),'actual_start_end_moments_Nm':np.asarray(moments)}
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    peak=model.known_force(tension(duration/2,1.,duration))
    r={'completed':failure is None,'duration_s':duration,'hz':240,'mesh_level':level,'tetrahedra':len(model.tets),'vertices':len(model.rest),'steps_completed':len(rows),'elapsed_s':time.monotonic()-start,'maximum_absolute_principal_stretch_strain':maximum_strain,'peak_nonrigid_coordinate_RMS_m':maximum_nonrigid,'minimum_J':minimum_J,'excitation_target_passed':failure is None and .02<=maximum_strain<=.08 and maximum_nonrigid>=.001,
       'maximum_original_residual_N':max((r['force_residual_N'] for r in rows),default=0.),'maximum_balance_minus_residual_work_J':max((abs(r['balance_error_J']-r['free_residual_work_J']) for r in rows),default=0.),'maximum_resultant_N':float(np.max(np.linalg.norm(resultants,axis=-1))) if resultants else 0.,'maximum_actual_moment_Nm':float(np.max(np.linalg.norm(moments,axis=-1))) if moments else 0.,'rest_peak_moment_Nm':np.cross(model.rest,peak).sum(0).tolist(),'solid_mass_kg':float(model.node_mass.sum()),'failure':failure,'trajectory':rec(path.with_suffix('.npz'))}
    dump(path.with_suffix('.json'),{'summary':r,'steps':rows})
    if timed_out is not None:raise timed_out
    return r,arrays


def fine_comparison(coarse,fine,duration):
    rc,a=coarse;rf,b=fine
    if not(rc['completed'] and rf['completed']):return {'passed':False,'reason':'Retainedfailedcapture'}
    delta=a['positions_m'][:,[1,4]]-b['positions_m'][:,[1,4]];error=float(np.sqrt(np.mean(delta**2)));signal=float(np.sqrt(np.mean((b['positions_m'][:,[1,4]]-b['rest_m'][[1,4]])**2)));tol=max(2e-6,.1*signal)
    noise={}
    for label,times in [('original_query_times',[.05,.1,.15,.2]),('drive_fraction_times',duration*np.array([.25,.5,.75,1.]))]:
        i=np.rint(np.array(times)*240).astype(int);e=a['positions_m'][i][:,[1,4]]-b['positions_m'][i][:,[1,4]];maximum=float(np.max(abs(e)))
        noise[label]={'times_s':np.asarray(times).tolist(),'coordinate_RMS_m':float(np.sqrt(np.mean(e**2))),'maximum_coordinate_error_m':maximum,'displacement20um_0p1sigma_passed':maximum<=2e-6,'depth200um_coordinateproxy_0p1sigma_passed':maximum<=20e-6}
    return {'coarse_tetrahedra':rc['tetrahedra'],'fine_tetrahedra':rf['tetrahedra'],'heldnode_full_trajectory_RMSE_m':error,'tolerance_m':tol,'passed':error<=tol,'noise_checks':noise}


def cost_plan(p,records):
    T=select_screen(records)
    if T is None:return {'selected_duration_s':None,'fine_authorized_by_design':False,'reason':'Neitherpredeclaredscreeneligible'}
    prior=json.loads(Path(p['geometry_only_prior_runtime_evidence']['path']).read_text());by={r['tetrahedra']:r for r in prior['captures']}
    forecast=2*sum(by[n]['elapsed_s']/by[n]['steps_completed']*round(T*240) for n in [1024,8192])+30
    return {'selected_duration_s':T,'prior_sparse_factor2_cost_forecast_s':forecast,'fine_hard_cap_s':600,'screen_case_runtime_s':[r['elapsed_s'] for r in records],'fine_authorized_by_design':forecast<=600,'force_amplitude_changed':False}


def execute(protocol,digest,output,stage,screen=None,screen_sha=None):
    p=validate(protocol,digest);torch.set_num_threads(2);out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    d={'schema':'revised-consistent-Gaussian-load-stage-v0.1','stage':stage,'completed':False,'protocol':{'path':str(protocol),'sha256':digest},'captures':[],'inverse_or_network_qualified':False,'temporal_validation_ran':False}
    cap=300 if stage=='screen' else 600
    def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError(f'Prospective{cap}sCPU2budget')));signal.alarm(cap)
    try:
        if stage=='screen':
            for T in p['screen']['durations_s']:
                r,_=capture(2,T,out/f'screen-T{round(1000*T)}ms');d['captures'].append(r);save()
            d['selection_and_cost']=cost_plan(p,d['captures']);d['completed']=len(d['captures'])==2 and all(r['completed'] for r in d['captures']);save()
        else:
            if not screen or sha(screen)!=screen_sha:raise ValueError('Frozenfullscreenrecordrequired')
            s=json.loads(Path(screen).read_text())
            if not s['completed'] or s['stage']!='screen' or s['protocol']['sha256']!=digest or len(s['captures'])!=2:raise ValueError('Completeprospectivematchedscreenrequired')
            selected=cost_plan(p,s['captures'])
            if selected!=s['selection_and_cost'] or not selected['fine_authorized_by_design']:raise ValueError('Nofineadmissionfrompredeclaredscreen')
            d['screen_summary']={'path':str(screen),'sha256':screen_sha};d['selection_and_cost']=selected;T=selected['selected_duration_s']
            chosen=next(r for r in s['captures'] if r['duration_s']==T)
            if sha(chosen['trajectory']['path'])!=chosen['trajectory']['sha256']:raise ValueError('Changedselectedscreen')
            with np.load(chosen['trajectory']['path'],allow_pickle=False) as a:old=(chosen,{k:a[k] for k in ['positions_m','rest_m']})
            d['spatial_comparisons']=[]
            probe=json.loads(Path(p['fixed_probe_protocol']['path']).read_text());P=np.concatenate([probe['volume_probes_m'],probe['surface_probes_m']]);previous=old
            for level in p['fine']['mesh_levels']:
                item=capture(level,T,out/f'fine{2*8**level}-T{round(1000*T)}ms');d['captures'].append(item[0]);d['spatial_comparisons'].append(fine_comparison(previous,item,T));save()
                if not item[0]['completed']:break
                # Fixed physical probe diagnostic between128 and bothfinemeshes.
                with np.load(chosen['trajectory']['path'],allow_pickle=False) as a:coarse=interpolate(a['rest_m'],a['tets'],a['positions_m'],P)[0]
                fine=interpolate(item[1]['rest_m'],item[1]['tets'],item[1]['positions_m'],P)[0]
                d.setdefault('fixed_probe_128_to_fine',{})[str(item[0]['tetrahedra'])]={'volume':error_metrics(coarse[:,:64],fine[:,:64],probe['volume_weights']),'surface':error_metrics(coarse[:,64:],fine[:,64:],probe['surface_area_weights'])};previous=item;save()
            d['completed']=len(d['captures'])==2 and all(r['completed'] for r in d['captures'])
            last=d['spatial_comparisons'][-1] if d['spatial_comparisons'] else {}
            d['spatial_RMS_development_gate_passed']=bool(last.get('passed',False))
            d['original_query_noise_checks_passed']=last.get('noise_checks',{}).get('original_query_times',{})
            d['whole_drive_noise_checks_passed']=last.get('noise_checks',{}).get('drive_fraction_times',{})
            d['finest_excitation_target_passed']=bool(d['captures'] and d['captures'][-1]['excitation_target_passed'])
            d['dynamic_forward_qualified']=False # No temporal qualification inthisstage
    except Exception as exc:
        d['execution_failure']=repr(exc)
        recorded={r['trajectory']['path'] for r in d['captures']}
        for candidate in sorted(out.glob('*.json')):
            if candidate.name=='summary.json':continue
            partial=json.loads(candidate.read_text()).get('summary')
            if partial and partial['trajectory']['path'] not in recorded:d['captures'].append(partial)
    finally:signal.alarm(0);save()
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--protocol',required=True)
    for name in ['screen','fine']:
        a=s.add_parser(name);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True);a.add_argument('--output',required=True)
        if name=='fine':a.add_argument('--screen-summary',required=True);a.add_argument('--screen-sha256',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        d=execute(a.protocol,a.protocol_sha256,a.output,a.action,getattr(a,'screen_summary',None),getattr(a,'screen_sha256',None));print(json.dumps({k:v for k,v in d.items() if k in ['completed','elapsed_s','selection_and_cost','spatial_RMS_development_gate_passed','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
