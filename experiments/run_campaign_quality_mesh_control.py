"""Matched quality-mesh-only control: old Gaussian quadrature/laws unchanged.

Physical integration loop copied transparently from frozen resolution driver;
only authored geometry constructor changes. No old file, force, fixture,
material, pulse, noise or gate is changed.
"""
from __future__ import annotations
import argparse,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import torch
try:
    from .campaign_quality_tet_refinement import geometry as quality_geometry
    from .campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian
    from .campaign_nonlinear_excitation import command,nonrigid_rms
    from .run_campaign_nonlinear_resolution import record,compare_mesh
    from .diagnose_nonlinear_physical_probes import interpolate,error_metrics,mass_properties
except ImportError:
    from campaign_quality_tet_refinement import geometry as quality_geometry
    from campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian
    from campaign_nonlinear_excitation import command,nonrigid_rms
    from run_campaign_nonlinear_resolution import record,compare_mesh
    from diagnose_nonlinear_physical_probes import interpolate,error_metrics,mass_properties


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def simulate_quality(level,hz,output):
    start=time.monotonic();model=SparseNonlinearGaussian(*quality_geometry(level))
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
    s['rest_load_centroids_m']=(model.load_weights@model.rest).tolist()
    s['rest_fixture_centroids_m']=(model.weights@model.rest).tolist()
    peakforce=model.known_force(command('balanced_axial',.1,1.))
    s['rest_peak_resultant_N']=peakforce.sum(0).tolist()
    s['rest_peak_moment_Nm']=np.cross(model.rest,peakforce).sum(0).tolist()
    s['mass_properties']=mass_properties(model.rest,model.tets)
    output.with_suffix('.json').write_text(json.dumps({'summary':s,'steps':rows},indent=2)+'\n')
    if failure and '300s' in failure['exception']:raise TimeoutError('300s hardcap; partial capture preserved')
    return s,arrays


def prepare(path):
    source=['experiments/run_campaign_quality_mesh_control.py','experiments/campaign_quality_tet_refinement.py','experiments/campaign_sparse_nonlinear_gaussian.py','experiments/run_campaign_nonlinear_resolution.py','experiments/campaign_nonlinear_excitation.py','experiments/diagnose_nonlinear_physical_probes.py','experiments/campaign_boundary_gaussians.py','experiments/campaign_compliant_gaussian_linear.py']
    old=Path('runs/gaussian-campaign-v01/nonlinear-resolution-v02')
    d={'schema':'prospective-quality-mesh-only-control-v0.1','status':'FROZEN_BEFORE_NEW_DYNAMIC_CALLS','CPU_threads':2,'wall_cap_s':300,'cases':[{'level':level,'tets':2*8**level,'hz':240} for level in [2,3,4]],
      'unchanged_physics':{'young_Pa':5000.,'poisson':.35,'density_kg_m3':625.,'fixture_scale':2.,'Gaussian_vertex_lumped_fixture_load_quadrature':'UNCHANGED','force':'OriginalCOMPRESSION balanced_axial,1Nperpatch,sin²0.2s,fixedworlddirections','damping':'OriginalunscaledD','duration_s':.2,'gravity':[0,0,0],'contact':'Unsupported','minimum_J':.1},
      'only_intervention':'Shortestcentral-octahedral-diagonalrefinementwithstablepublicgeometryties;original5verticespreserved',
      'old_captures':{str(n):record(old/f'mesh{n}-hz240.npz') for n in [128,1024,8192]},'fixed_probe_protocol':record('runs/operator-dev/nonlinear-physical-probes-v01/protocol.json'),
      'metrics':['Originalnodes1/4fulltrajectory10%/2umgateunchanged','Sparsequery2um/20umnoisechecksunchanged','Fixedphysical64volume+6surfaceprobes,rawsame-countoldvsnew','Exactmass/inertia,oldlumpedGaussianloadandfixturecentroids,resultants/moments','All2–8%/1mmexcitationfailuresretained'],
      'no_temporal_inverse_or_training':True,'source_sha256':{p:sha(p) for p in source}}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(d,indent=2)+'\n');return d


def run(protocol,digest,output):
    if sha(protocol)!=digest:raise ValueError('Changedqualityprotocol')
    p=json.loads(Path(protocol).read_text())
    for path,expected in p['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Changedsource '+path)
    for r in [*p['old_captures'].values(),p['fixed_probe_protocol']]:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changedevidence '+r['path'])
    probes=json.loads(Path(p['fixed_probe_protocol']['path']).read_text());P=np.concatenate([probes[g+'_probes_m'] for g in ['volume','surface','corner']])
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    d={'schema':'quality-mesh-only-control-result-v0.1','completed':False,'protocol':{'path':str(protocol),'sha256':digest},'captures':[],'new_mesh_comparisons':[],'same_count_old_vs_quality_probes':{},'old_failed_gates_unchanged':True,'inverse_or_training_qualified':False}
    def save():d['elapsed_s']=time.monotonic()-start;(output/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Prospective300sCPU2budget')));signal.alarm(300)
    try:
        previous=None
        for case in p['cases']:
            current=simulate_quality(case['level'],240,output/f"quality{case['tets']}-hz240");d['captures'].append(current[0]);save()
            if not current[0]['completed']:break
            if previous:d['new_mesh_comparisons'].append(compare_mesh(previous,current));save()
            previous=current
            with np.load(p['old_captures'][str(case['tets'])]['path'],allow_pickle=False) as a:oldfield=interpolate(a['rest_m'],a['tets'],a['positions_m'],P)[0]
            newfield=interpolate(current[1]['rest_m'],current[1]['tets'],current[1]['positions_m'],P)[0]
            d['same_count_old_vs_quality_probes'][str(case['tets'])]={g:error_metrics(oldfield[:,sl],newfield[:,sl],w) for g,sl,w in [('volume',slice(0,64),probes['volume_weights']),('surface',slice(64,70),probes['surface_area_weights']),('corner',slice(70,72),probes['corner_weights'])]};save()
        d['completed']=len(d['captures'])==3 and all(r['completed'] for r in d['captures'])
        d['new_family_spatial_development_gate_passed']=bool(d['new_mesh_comparisons']) and d['new_mesh_comparisons'][-1]['passed']
        d['limits']=['MeshfamilyONLYmatchedintervention;knownloadandfixturequadratureunchanged','No centralalignment/tension/width/slowerpulseablationyet','Passing10%doesnotqualifynoise-scalesensorinference','No physicalcausalattributionfromonetestalone','Alloldfailedgateobjectsremainunchanged']
    except Exception as exc:
        d['execution_failure']=repr(exc)
        recorded={r['trajectory']['path'] for r in d['captures']}
        for candidate in sorted(output.glob('quality*.json')):
            partial=json.loads(candidate.read_text())['summary']
            if partial['trajectory']['path'] not in recorded:d['captures'].append(partial)

    finally:signal.alarm(0);save()
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--protocol',required=True)
    a=s.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
    else:
        r=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:r.get(k) for k in ['completed','elapsed_s','new_family_spatial_development_gate_passed','execution_failure']},indent=2));raise SystemExit(0 if r['completed'] else 1)
