"""Frozen lower-load follow-up with eight prospectively new material/support settings.

Preserves v02 negative. All5 development settings plus8 newly drawn settings retained. Independent DOP853
validates temporal algebra; no spatial/nonlinear/calibration claim follows.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,time
import numpy as np
import torch
from experiments import campaign_clamped_deformation_v02 as model
from experiments.run_clamped_deformation_v01 import volume_metrics,write_new
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='runs/operator-dev/clamped-deformation-v03/protocol.json'
OUT='runs/gaussian-campaign-v01/clamped-deformation-v03'
SOURCES=['experiments/campaign_clamped_deformation_v02.py','tests/test_clamped_deformation_v02.py',
         'experiments/run_clamped_deformation_v02.py','experiments/run_clamped_deformation_v03.py','experiments/supervise_clamped_deformation_v03.py','tests/test_clamped_followup_v03.py',
         'experiments/supervise_surrogate_sbc_continuation.py','experiments/run_clamped_deformation_v01.py',
         'experiments/campaign_clamped_deformation_v01.py','experiments/campaign_linear_fem.py',
         'experiments/campaign_boundary_gaussians.py','experiments/campaign_p2_keast_quadrature.py',
         'experiments/diagnose_c38_supported_motion.py','tests/test_clamped_qualification_v01.py']
from scipy.stats import qmc
NEW_UNITS=qmc.LatinHypercube(d=2,seed=2026102431).random(8)
DESIGN=dict(force_scale=.8,new_settings_seed=2026102431,new_settings=8,development_settings=5,design_tuned_on_v02=True,design_reason='Reduceallloads20percent after v02 maximumrecordedstrain .0512 exceeded.05; exactlinearity predicts.041margin. Priornegativepreserved; new8settingsnotseenbeforefreeze',objects=['lowE_lowK','lowE_highK','highE_lowK','highE_highK','geometric_center']+[f'new_LHS_{i:02d}'for i in range(8)],
            units=[[0.,0.],[0.,1.],[1.,0.],[1.,1.],[.5,.5]]+NEW_UNITS.tolist(),loads=['bending','stretching'],
            late_clocks_s=[1.6,1.8,2.],
            deformation_metric='PositiveKeast24 volume quadrature and proper weighted rigidfit; coordinateRMS aggregate lateclocks, not equilibrium',
            strain_metric='Maximumabsolute infinitesimalstrain tensorcomponent and minimumJ overALL41 OUTPUTclocks/elements, notcontinuous-in-time extremum',
            independent_comparison='SeparateNumPyassembly plusDOP853, maxabsolutepositiondifference<=2um andvelocity<=10um/s overALL41clocks/nodes/coordinates',
            maximum_independent_velocity_error_m_s=10e-6,maximum_free_residual_N=1e-9,
            all_cases_required=True,no_case_discard=True,no_retry=True,
            inference=False,policy_rerun=False,spatial_convergence=False,nonlinear_validation=False,
            resources=dict(CPU=[14],threads=1,wall_s=600,RSS_bytes=2*1024**3,no_GPU=True,disk_reserve_bytes=30*1024**3))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    if (ROOT/OUT).exists():raise FileExistsError('No output reuse')
    old=ROOT/'runs/gaussian-campaign-v01/clamped-deformation-v02/summary.json';s=json.loads(old.read_text())
    if not s['completed']or s['all_gates_passed']or s['temporal_algebra_passed_cases']!=10 or s['deformation_passed_cases']!=9:raise ValueError('Preserve v02 completed9of10 strain negative')
    write_new(ROOT/PROTOCOL,dict(design=DESIGN,model_design=model.design(),source_sha256={n:sha(ROOT/n)for n in SOURCES},
                              predecessor=dict(path=str(old.relative_to(ROOT)),sha256=sha(old),negative_preserved=True),output=OUT,
                              status='FROZEN_BEFORE_PRODUCTION_SOLVES',raw_LOCAL=True))
    return sha(ROOT/PROTOCOL)


def run(digest):
    p=json.loads((ROOT/PROTOCOL).read_text())
    if sha(ROOT/PROTOCOL)!=digest or p['design']!=DESIGN or p['model_design']!=model.design()or p['source_sha256']!={n:sha(ROOT/n)for n in SOURCES}:raise ValueError('Frozen source/design/protocol')
    if sha(ROOT/p['predecessor']['path'])!=p['predecessor']['sha256']:raise ValueError('Changed predecessor')
    if shutil.disk_usage(ROOT).free<DESIGN['resources']['disk_reserve_bytes']+1024**3:raise OSError('Disk reserve')
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(DESIGN['resources']['CPU']))
    torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.monotonic();out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    write_new(out/'owner.json',dict(pid=os.getpid(),protocol_sha256=digest,no_GPU=True));f=model.ExactClampedForward();theta=model.physical(np.array(DESIGN['units']));forces=DESIGN['force_scale']*np.stack([model.programs()[n]for n in DESIGN['loads']])
    exact=f.trajectory(theta,forces,return_state=True);u=exact['displacement'];v=exact['velocity'];np.savez(out/'exact-state-LOCAL.npz',**exact)
    rows=[];g=model.GATES
    for i,name in enumerate(DESIGN['objects']):
        independent=model.independent_dop853(theta[i],forces,return_state=True);np.savez(out/f'independent-{i}-LOCAL.npz',displacement=independent['displacement'],velocity=independent['velocity'])
        for j,load in enumerate(DESIGN['loads']):
            ids=np.rint(np.array(DESIGN['late_clocks_s'])/model.OUTPUT_H).astype(int);metrics=volume_metrics(f.X,f.tets,u[i,j,ids]);gradient=model.displacement_gradient(f,u[i,j]);strain=.5*(gradient+gradient.transpose(0,1,3,2))
            maxstrain=float(np.max(abs(strain)));minJ=float(np.linalg.det(np.eye(3)+gradient).min());position_error=float(np.max(abs(u[i,j]-independent['displacement'][j])));velocity_error=float(np.max(abs(v[i,j]-independent['velocity'][j])))
            deformation_pass=bool(metrics['volume_nonrigid_coordinate_RMS_m']>=g['minimum_nonrigid_RMS_m']and metrics['nonrigid_to_total_RMS_ratio']>=g['minimum_nonrigid_fraction']and maxstrain<=g['maximum_linear_strain']and minJ>=g['minimum_J'])
            residual=float(np.max(exact['maximum_free_residual_N'][i]));temporal_pass=bool(position_error<=g['maximum_independent_trajectory_error_m']and velocity_error<=DESIGN['maximum_independent_velocity_error_m_s']and residual<=DESIGN['maximum_free_residual_N'])
            rows.append(dict(object_index=i,object=name,load=load,**metrics,maximum_output_clock_abs_strain_component=maxstrain,minimum_output_clock_J=minJ,
                             independent_max_position_error_m=position_error,independent_max_velocity_error_m_s=velocity_error,maximum_free_residual_N=residual,
                             deformation_passed=deformation_pass,temporal_algebra_passed=temporal_pass,passed=deformation_pass and temporal_pass))
        write_new(out/f'progress-{i}.json',dict(object_index=i,completed=True,elapsed_s=time.monotonic()-started,independent_solver=independent['diagnostics']))
    report=dict(new_settings_passed_cases=sum(r['passed']for r in rows if r['object_index']>=5),development_settings_passed_cases=sum(r['passed']for r in rows if r['object_index']<5),completed=True,protocol_sha256=digest,objects=13,cases=26,development_cases=10,newly_held_settings_cases=16,all_cases_retained=True,rows=rows,deformation_passed_cases=sum(r['deformation_passed']for r in rows),
                temporal_algebra_passed_cases=sum(r['temporal_algebra_passed']for r in rows),all_gates_passed=all(r['passed']for r in rows),elapsed_s=time.monotonic()-started,
                observation_noise_std_m=model.SIGMA,source_sha256=p['source_sha256'],previous_BE_negative_preserved=True,previous_v02_strain_negative_preserved=True,force_scale=DESIGN['force_scale'],design_tuned_on_v02=True,new_settings_not_used_for_design=True,
                policy_comparison_run=False,inference_run=False,raw_states_LOCAL=True,continuum_or_nonlinear_validation=False,
                scope='Five tuning/development settings plus eight prospectively fixed newLHSsettings; finite linear model, notcontinuum orpopulation calibration')
    if p['source_sha256']!={n:sha(ROOT/n)for n in SOURCES}:raise ValueError('Source changed during run')
    write_new(out/'summary.json',report);print(json.dumps({k:report[k]for k in ['completed','deformation_passed_cases','temporal_algebra_passed_cases','all_gates_passed','elapsed_s']}))
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--digest');v=a.parse_args();print(prepare())if v.mode=='prepare'else run(v.digest)
