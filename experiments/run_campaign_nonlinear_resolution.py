"""Bounded prospective sparse nonlinear mesh/time qualification, no inverse fit."""
from __future__ import annotations
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import torch
try:
    from .campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian
    from .campaign_nonlinear_excitation import command,nonrigid_rms
except ImportError:
    from campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian
    from campaign_nonlinear_excitation import command,nonrigid_rms


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def record(p):return {'path':str(p),'sha256':sha(p)}

def prepare(path):
    dependencies=['experiments/campaign_sparse_nonlinear_gaussian.py','experiments/run_campaign_nonlinear_resolution.py','experiments/campaign_nonlinear_excitation.py','experiments/campaign_nonlinear_gaussian_reference.py','experiments/campaign_nonlinear_fem.py','experiments/campaign_boundary_gaussians.py','experiments/campaign_compliant_gaussian_linear.py','experiments/campaign_material_gaussians.py','experiments/campaign_linear_fem.py']
    d={'schema':'nonlinear-sparse-resolution-development-v0.2','status':'FROZEN_BEFORE_RESOLUTION_CALLS','CPU_threads':2,'wall_cap_s':900,
       'physical_parameters':{'young_Pa':5000.,'poisson':.35,'density_kg_m3':625.,'fixture_scale':2.,'damping':'Unscaled original candidateD','force_pattern':'balanced_axial','amplitude_N_per_patch':1.,'duration_s':.2,'minimum_J':.1,'contact':'Unsupported','gravity':[0,0,0]},
       'mesh':{'levels':[2,3,4],'tet_counts':[128,1024,8192],'hz':240,'stop_after_first_adjacent_pass':True,'selected_mesh':'Fine mesh of first passing adjacent comparison','held_nodes':[1,4],'relative_full_trajectory_RMSE_tolerance':.1,'absolute_floor_m':2e-6},
       'temporal':{'rates_hz':[240,480,960],'conditional_final_rate_hz':1920,'conditional_trigger':'Finest960 adjacent comparison fails originalgate or contraction','qualified_mesh_only':True,'relative_full_trajectory_RMSE_tolerance_to_peak_nonrigid':.05,'absolute_floor_m':2e-6},
       'sparse_query_numerical_accuracy':{'nodes':[1,4],'times_s':[.05,.1,.15,.2],'noise_scales_m':{'displacement_diagnostic':20e-6,'existing_depth_scale_coordinate_proxy':200e-6},'fraction_of_sigma_limit':.1,'gate_uses_maximum_absolute_coordinate_error':True,'depth_note':'Coordinate discrepancy at depthnoise scale, NOT renderedrangeerror'},
       'excitation_target':{'absolute_principal_stretch_strain':[.02,.08],'peak_best_rigid_removed_coordinate_RMS_m_at_least':.001,'no_amplitude_retuning':True},
       'torch128_parity':{'prior_protocol':record('runs/operator-dev/nonlinear-excitation-v01/protocol.json'),'trajectory':record('runs/gaussian-campaign-v01/nonlinear-excitation-v01/mesh128-balanced_axial.npz'),'maximum_absolute_position_error_m':2e-9},
       'source_sha256':{p:sha(p) for p in dependencies},'limits':['Allfailedresolutionattemptsretained','Passing10%developmentgate DOESNOTqualify20um-noiseinference','No inverse, networks, cohortdata or GPU','Initialphase1failuresimmutable']}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(d,indent=2)+'\n');return d


def simulate(level,hz,output):
    start=time.monotonic();model=SparseNonlinearGaussian.from_design(level)
    x=model.rest.copy();v=np.zeros_like(x);target=np.repeat(x[None],2,axis=0);h=1/hz
    qs=[x.copy()];vs=[v.copy()];rows=[];forces=[];moments=[];held=[]
    minJ=1.;peakstrain=0.;peaknonrigid=0.;failure=None
    try:
        for k in range(round(.2*hz)):
            controls=command('balanced_axial',(k+.5)*h,1.);f=model.known_force(controls);old=x.copy()
            x,v,row=model.step(x,v,target,target,f,h)
            F=np.eye(3)+np.einsum('eai,eaJ->eiJ',(x-model.rest)[model.tets],model.gradients)
            peakstrain=max(peakstrain,float(np.max(abs(np.linalg.svd(F,compute_uv=False)-1))))
            peaknonrigid=max(peaknonrigid,nonrigid_rms(model.rest,x));minJ=min(minJ,row['minimum_J'])
            qs.append(x.copy());vs.append(v.copy());forces.append(f);held.append(controls)
            moments.append(np.stack([np.cross(old,f).sum(0),np.cross(x,f).sum(0)]))
            rows.append(row)
    except Exception as exc:failure={'exception':repr(exc),'history':getattr(exc,'history',[]),'step_attempted':len(rows)+1}
    arrays={'positions_m':np.asarray(qs),'velocities_m_s':np.asarray(vs),'times_s':np.arange(len(qs))/hz,'rest_m':model.rest,'tets':model.tets,'known_nodal_forces_N':np.asarray(forces),'held_patch_forces_N':np.asarray(held),'actual_start_end_moments_Nm':np.asarray(moments)}
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    s={'completed':failure is None,'mesh_level':level,'tetrahedra':len(model.tets),'vertices':len(model.rest),'hz':hz,'steps_completed':len(rows),'elapsed_s':time.monotonic()-start,'minimum_J':minJ,'maximum_absolute_principal_stretch_strain':peakstrain,'peak_nonrigid_coordinate_RMS_m':peaknonrigid,'excitation_target_passed':failure is None and .02<=peakstrain<=.08 and peaknonrigid>=.001,
       'maximum_original_residual_N':max((r['force_residual_N'] for r in rows),default=0.),'maximum_balance_minus_residual_work_J':max((abs(r['balance_error_J']-r['free_residual_work_J']) for r in rows),default=0.),'algorithmic_shift_N_m':0.,'fallback_directions':sum(h.get('direction')=='mass_fixture_metric_descent' for r in rows for h in r['history']),
       'maximum_resultant_norm_N':float(np.max(np.linalg.norm(np.sum(forces,axis=1),axis=-1))) if forces else 0.,'maximum_actual_moment_Nm':float(np.max(np.linalg.norm(moments,axis=-1))) if moments else 0.,'solid_mass_kg':float(model.node_mass.sum()),'failure':failure,'trajectory':record(output.with_suffix('.npz'))}
    output.with_suffix('.json').write_text(json.dumps({'summary':s,'steps':rows},indent=2)+'\n')
    if failure and '900s' in failure['exception']:raise TimeoutError('900s hardcap; partial capture preserved')
    return s,arrays


def noise_checks(coarse,fine,hz_coarse,hz_fine):
    ic=np.rint(np.array([.05,.1,.15,.2])*hz_coarse).astype(int);jf=np.rint(np.array([.05,.1,.15,.2])*hz_fine).astype(int)
    d=coarse['positions_m'][ic][:,[1,4]]-fine['positions_m'][jf][:,[1,4]]
    rms=float(np.sqrt(np.mean(d*d)));maximum=float(np.max(abs(d)))
    return {'query_coordinate_RMSE_m':rms,'query_maximum_absolute_error_m':maximum,'displacement20um_0p1sigma':{'threshold_m':2e-6,'passed':maximum<=2e-6},'depth200um_coordinateproxy_0p1sigma':{'threshold_m':20e-6,'passed':maximum<=20e-6},'scope':'Displacementcoordinatesatpublicqueries; secondscaleisNOTactualdeptherror'}


def compare_mesh(a,b):
    sa,aa=a;sb,ab=b
    if not(sa['completed'] and sb['completed']):return {'passed':False,'reason':'Preserved failed mesh capture'}
    d=aa['positions_m'][:,[1,4]]-ab['positions_m'][:,[1,4]];error=float(np.sqrt(np.mean(d*d)))
    motion=float(np.sqrt(np.mean((ab['positions_m'][:,[1,4]]-ab['rest_m'][[1,4]])**2)));tol=max(2e-6,.1*motion)
    return {'coarse_tetrahedra':sa['tetrahedra'],'fine_tetrahedra':sb['tetrahedra'],'heldnode_full_trajectory_RMSE_m':error,'fine_held_displacement_RMS_m':motion,'tolerance_m':tol,'passed':error<=tol,'noise_referenced':noise_checks(aa,ab,240,240)}


def compare_time(a,b):
    sa,aa=a;sb,ab=b
    if not(sa['completed'] and sb['completed']):return {'passed':False,'reason':'Preserved failed temporal capture'}
    ratio=sb['hz']//sa['hz'];error=float(np.sqrt(np.mean((aa['positions_m']-ab['positions_m'][::ratio])**2)));tol=max(2e-6,.05*sb['peak_nonrigid_coordinate_RMS_m'])
    return {'rates_hz':[sa['hz'],sb['hz']],'full_trajectory_RMSE_m':error,'tolerance_m':tol,'accuracy_passed':error<=tol,'noise_referenced':noise_checks(aa,ab,sa['hz'],sb['hz'])}


def run(protocol_path,digest,output):
    if sha(protocol_path)!=digest:raise ValueError('Changed protocol')
    p=json.loads(Path(protocol_path).read_text())
    for path,expected in p['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Changed source '+path)
    for r in p['torch128_parity'].values():
        if isinstance(r,dict) and sha(r['path'])!=r['sha256']:raise ValueError('Changed priorparityevidence')
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    d={'schema':'sparse-nonlinear-resolution-result-v0.2','completed':False,'protocol':{'path':str(protocol_path),'sha256':digest},'captures':[],'mesh_comparisons':[],'time_comparisons':[],'qualified_mesh_tetrahedra':None,'qualified_rate_hz':None,'inverse_qualified':False,'neural_training_authorized':False}
    def save():d['elapsed_s']=time.monotonic()-start;(output/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
    def alarm(signum,frame):raise TimeoutError('Prospective900sCPU2budgetexhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        previous=None;selected=None
        for level in p['mesh']['levels']:
            current=simulate(level,240,output/f'mesh{2*8**level}-hz240');d['captures'].append(current[0]);save()
            if level==2 and current[0]['completed']:
                with np.load(p['torch128_parity']['trajectory']['path'],allow_pickle=False) as prior:
                    err=float(np.max(abs(prior['positions_m']-current[1]['positions_m'])))
                d['torch128_full_trajectory_parity']={'maximum_absolute_position_error_m':err,'tolerance_m':2e-9,'passed':err<=2e-9};save()
            if previous:
                comparison=compare_mesh(previous,current);d['mesh_comparisons'].append(comparison);save()
                if comparison['passed']:selected=current;d['qualified_mesh_tetrahedra']=current[0]['tetrahedra'];break
            if not current[0]['completed']:break
            previous=current
        if selected:
            levels=[selected]
            for hz in (480,960):
                item=simulate(selected[0]['mesh_level'],hz,output/f"mesh{selected[0]['tetrahedra']}-hz{hz}");levels.append(item);d['captures'].append(item[0]);d['time_comparisons'].append(compare_time(levels[-2],item));save()
                if not item[0]['completed']:break
            if len(d['time_comparisons'])>=2 and all('full_trajectory_RMSE_m' in r for r in d['time_comparisons']):
                e=d['time_comparisons'];passed=e[-1]['accuracy_passed'] and e[-1]['full_trajectory_RMSE_m']<e[-2]['full_trajectory_RMSE_m']
                if not passed:
                    item=simulate(selected[0]['mesh_level'],1920,output/f"mesh{selected[0]['tetrahedra']}-hz1920");d['captures'].append(item[0]);comparison=compare_time(levels[-1],item);d['time_comparisons'].append(comparison);levels.append(item);save()
                    passed=comparison.get('accuracy_passed',False) and comparison['full_trajectory_RMSE_m']<e[-2]['full_trajectory_RMSE_m']
                if passed:d['qualified_rate_hz']=levels[-1][0]['hz']
            d['finest_excitation_target_passed']=levels[-1][0]['excitation_target_passed']
        d['completed']=True
        d['development_field_resolution_passed']=d['qualified_mesh_tetrahedra'] is not None and d['qualified_rate_hz'] is not None and d.get('torch128_full_trajectory_parity',{}).get('passed',False)
        d['limits']=['Mesh/timechecksarefactorized;nohiddenamplituderetuning','10%developmentaccuracydoesnotqualify20um-noiseinverse','Noisechecksarecoordinates,notrenderedrangeerrors','No nonlinear inverse or training ran; all earlierfailures preserved']
    except Exception as exc:
        d['execution_failure']={'exception':repr(exc),'history':getattr(exc,'history',[])}
        # A budget interruption writes its partial capture before rethrowing.
        recorded={r['trajectory']['path'] for r in d['captures']}
        for candidate in sorted(output.glob('mesh*.json')):
            partial=json.loads(candidate.read_text())['summary']
            if partial['trajectory']['path'] not in recorded:d['captures'].append(partial)

    finally:signal.alarm(0);save()
    return d

if __name__=='__main__':
    parser=argparse.ArgumentParser();s=parser.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--protocol',required=True)
    a=s.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True);a.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        result=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:v for k,v in result.items() if k in ['completed','elapsed_s','qualified_mesh_tetrahedra','qualified_rate_hz','development_field_resolution_passed','execution_failure']},indent=2));raise SystemExit(0 if result['completed'] else 1)
