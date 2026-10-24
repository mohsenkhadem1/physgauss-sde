"""Prospective all-corner clamped deformation qualification, raw states local.

This is finite-model development, not a nonlinear/continuum dataset admission.
The protocol is exclusive and hash-binds source before any production solve.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, time
import numpy as np
import torch
from experiments import campaign_clamped_deformation_v01 as model
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.diagnose_c38_supported_motion import motion_components

ROOT=Path(__file__).resolve().parents[1]
OUT='runs/gaussian-campaign-v01/clamped-deformation-v01'
PROTOCOL='runs/operator-dev/clamped-deformation-v01/protocol.json'
SOURCES=['experiments/supervise_clamped_deformation_v01.py','experiments/supervise_surrogate_sbc_continuation.py','experiments/run_clamped_deformation_v01.py','tests/test_clamped_qualification_v01.py',
         'experiments/campaign_clamped_deformation_v01.py','tests/test_clamped_deformation_v01.py',
         'experiments/campaign_linear_fem.py','experiments/campaign_boundary_gaussians.py',
         'experiments/campaign_p2_keast_quadrature.py','experiments/diagnose_c38_supported_motion.py']
DESIGN=dict(objects=['lowE_lowK','lowE_highK','highE_lowK','highE_highK','geometric_center'],
            units=[[0.,0.],[0.,1.],[1.,0.],[1.,1.],[.5,.5]],loads=['bending','stretching'],
            late_clocks_s=[1.6,1.8,2.],
            deformation_metric='Positive Keast24 volume quadrature and proper weighted rigid registration; coordinate RMS',
            deformation_scope='Aggregate of three specified late clocks (not claimed equilibrium), separately for every object/load',
            strain_metric='Maximum absolute infinitesimal strain tensor component over all elements and all recorded times',
            time_metric='All 20Hz clocks, aggregate nodal-time coordinate RMS successive timestep differences; p>=1 and conservative 2*lastdifference<=20um conditional position indicator; no velocity or continuum claim',
            require_every_case=True,allow_retry=False,no_case_discard=True,
            inference=False,spatial_convergence=False,nonlinear_validation=False,
            resources=dict(CPU=[14],threads=1,wall_s=300,RSS_bytes=2*1024**3,disk_reserve_bytes=30*1024**3,no_GPU=True))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def volume_metrics(X,tets,displacements):
    bary,w=positive_keast24();vol=np.linalg.det((X[tets[:,1:]]-X[tets[:,:1]]).transpose(0,2,1))/6
    if np.any(vol<=0):raise ValueError('Positive material volumes required')
    rest=np.einsum('qa,eai->eqi',bary,X[tets]).reshape(-1,3);weights=(vol[:,None]*w).ravel();rows=[]
    for u in displacements:
        deformed=np.einsum('qa,eai->eqi',bary,(X+u)[tets]).reshape(-1,3)
        rows.append(motion_components(rest,deformed,weights))
    total=float(np.sqrt(np.mean([r['total_coordinate_motion_RMS_m']**2 for r in rows])))
    nr=float(np.sqrt(np.mean([r['nonrigid_coordinate_RMS_m']**2 for r in rows])))
    return dict(volume_motion_coordinate_RMS_m=total,volume_nonrigid_coordinate_RMS_m=nr,nonrigid_to_total_RMS_ratio=nr/total if total else 0.)


def prepare():
    p=ROOT/PROTOCOL
    if (ROOT/OUT).exists():raise FileExistsError('No retry/output reuse')
    write_new(p,dict(design=DESIGN,model_design=model.design(),source_sha256={n:sha(ROOT/n)for n in SOURCES},
                     output=OUT,status='FROZEN_BEFORE_PRODUCTION_SOLVES',raw_LOCAL=True))
    return sha(p)


def run(digest):
    p=json.loads((ROOT/PROTOCOL).read_text())
    if sha(ROOT/PROTOCOL)!=digest or p['design']!=DESIGN or p['model_design']!=model.design() or p['source_sha256']!={n:sha(ROOT/n)for n in SOURCES}:raise ValueError('Changed frozen protocol/source')
    if shutil.disk_usage(ROOT).free<DESIGN['resources']['disk_reserve_bytes']+1024**3:raise OSError('Disk reserve')
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(DESIGN['resources']['CPU']))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.monotonic();out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False)
    write_new(out/'owner.json',dict(pid=os.getpid(),protocol_sha256=digest,no_GPU=True))
    f=model.ClampedForward();theta=model.physical(np.array(DESIGN['units']));saved=[]
    for h in model.TIME_GATES['h_values']:
        forces=np.stack([model.programs(h)['qualification'][n]for n in DESIGN['loads']])
        u=f.trajectory(theta,forces,h);saved.append(u)
        np.save(out/f'trajectories-h{round(1/h)}-LOCAL.npy',u)
        if time.monotonic()-started>DESIGN['resources']['wall_s']:raise TimeoutError('No automatic retry')
    hs=model.TIME_GATES['h_values'];sub=[u[:,:,::round(1/(20*h))]for u,h in zip(saved,hs)]
    d01=np.sqrt(np.mean((sub[0]-sub[1])**2,axis=(2,3,4)));d12=np.sqrt(np.mean((sub[1]-sub[2])**2,axis=(2,3,4)))
    rows=[];g=model.DEFORMATION_GATES;tg=model.TIME_GATES
    for i,name in enumerate(DESIGN['objects']):
        for j,load in enumerate(DESIGN['loads']):
            u=saved[-1][i,j];ids=np.rint(np.array(DESIGN['late_clocks_s'])/hs[-1]).astype(int)
            metrics=volume_metrics(f.X,f.tets,u[ids]);strain=model.deformation_metrics(f,u)
            maxstrain=float(np.max(strain['maximum_linear_strain']));minJ=float(np.min(strain['minimum_J']))
            deformation_pass=bool(metrics['volume_nonrigid_coordinate_RMS_m']>=g['minimum_nonrigid_RMS_m'] and metrics['nonrigid_to_total_RMS_ratio']>=g['minimum_nonrigid_fraction'] and maxstrain<=g['maximum_linear_strain'] and minJ>=g['minimum_J'])
            contraction=float(d12[i,j]/d01[i,j])if d01[i,j]>0 else (0. if d12[i,j]==0 else None)
            time_pass=bool(d12[i,j]<=tg['maximum_last_pair_RMS_m'] and contraction is not None and contraction<=tg['maximum_pair_contraction'])
            rows.append(dict(object_index=i,object=name,load=load,**metrics,maximum_abs_linear_strain_component=maxstrain,minimum_J=minJ,
                             timestep_first_difference_RMS_m=float(d01[i,j]),timestep_last_difference_RMS_m=float(d12[i,j]),timestep_contraction=contraction,observed_temporal_order=float(-np.log2(contraction))if contraction is not None and contraction>0 else None,conditional_position_indicator_m=float(2*d12[i,j]),
                             deformation_passed=deformation_pass,timestep_passed=time_pass,passed=deformation_pass and time_pass))
    report=dict(completed=True,protocol_sha256=digest,objects=5,cases=10,all_cases_retained=True,rows=rows,
                deformation_passed_cases=sum(r['deformation_passed']for r in rows),timestep_passed_cases=sum(r['timestep_passed']for r in rows),
                all_gates_passed=all(r['passed']for r in rows),elapsed_s=time.monotonic()-started,
                observation_noise_std_m=model.SIGMA,source_sha256=p['source_sha256'],
                policy_comparison_run=False,raw_states_LOCAL=True,continuum_or_nonlinear_validation=False,
                scope='Five prospective finite-model development corners/interior; not a population result')
    if p['source_sha256']!={n:sha(ROOT/n)for n in SOURCES}:raise ValueError('Source changed during run')
    write_new(out/'summary.json',report);print(json.dumps({k:report[k]for k in ['completed','deformation_passed_cases','timestep_passed_cases','all_gates_passed','elapsed_s']}))
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--digest');v=a.parse_args()
    print(prepare())if v.mode=='prepare'else run(v.digest)
