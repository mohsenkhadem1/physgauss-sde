"""Audit every declared fitted slot; score only complete, sealed forecasts.

This reads targets only after calibration seals. The cohort is development and
uses coarse references: results must not be presented as held OOD evidence.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.audit_development import shape_rms
from scripts.complex_scenes_v18.run_development import ROOT,write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(actual,predicted):
    error=predicted-actual
    shape=np.array([[shape_rms(a,p)for a,p in zip(aa,pp)]for aa,pp in zip(actual,predicted)])
    return dict(vector_rms_mm=float(1000*np.sqrt(np.mean(np.sum(error**2,axis=-1)))),rigid_aligned_shape_rms_mm=float(1000*np.sqrt(np.mean(shape**2))))


def audit(output,material_label='material-scene-fit-01'):
    base=ROOT/'runs/complex-scenes-v18';cases=json.loads((base/'development-prefixes/manifest.json').read_text())['cases']
    labels={'ours':'material-scene-fit-01','analytic-control':'analytic-scene-fit-01','spring-gaus':'spring-edge-fit-01','phystwin':'phystwin-edge-fit-01','uniphy':'uniphy-screened-adjoint-development'};rows=[]
    if Path(material_label).name!=material_label:raise ValueError('Material label basename required')
    labels['ours']=material_label
    for case in cases:
        name=f"{case['case']}-v{case['variant']}"
        for method,label in labels.items():
            row=dict(case=case['case'],variant=case['variant'],method=method,state='pending')
            rows.append(row)
            if method=='uniphy'and case['case']!='container_drop':
                row.update(state='unsupported',reason='Released demo adapter currently supports gravity drops with fixed bowl/sphere geometry; other common boundary types are not implemented')
                continue
            folder=base/label/name
            repaired=base/'phystwin-edge-fit-memoryfix-01'/name
            if method=='phystwin'and case['case']=='container_drop'and repaired.exists():
                previous=folder/'fit-seal.json'
                row.update(replacement_reason='Rerun after native autograd output-retention repair',previous_fit_seal_sha256=sha(previous)if previous.exists()else None)
                folder=repaired
            row['fit_directory']=str(folder.relative_to(ROOT))
            sealpath=folder/'fit-seal.json'
            if not sealpath.exists():continue
            seal=json.loads(sealpath.read_text());row['fit_seal_sha256']=sha(sealpath)
            if not seal['completed']:row.update(state='fit_failed');continue
            if seal['forecast_targets_read']or seal['observations_sha256']!=case['observations_sha256']:raise ValueError('Invalid calibration seal')
            if method in ('ours','analytic-control'):
                raw=folder/'sealed-rollout'/name;donepath=raw/'complete.json';predpath=raw/'trajectory.npz'
                row['calibration_vector_rms_mm']=1000*np.sqrt(seal['best']['loss_m2'])
                row['fit_evaluations']=len(seal['trials']);row['optimizer_converged']=seal['optimizer_converged']
            else:
                donepath=folder/'rollout-complete.json';predpath=folder/'sealed-rollout.npz'
                if sha(folder/'best.pt')!=seal['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                row['calibration_vector_rms_mm']=float(np.sqrt(seal['best_loss_mm2']))
                row['fit_evaluations']=sum(r.get('completed',False)for r in seal['history'])
                row['requested_epochs_completed']=seal.get('requested_epochs_completed')
            if not donepath.exists():row['state']='forecast_pending';continue
            done=json.loads(donepath.read_text())
            if not done['completed']:row.update(state='forecast_failed',error=done.get('error'));continue
            if method in ('ours','analytic-control'):
                if done['parameters']!=seal['best']['parameters']:raise ValueError('Forecast parameters changed')
            elif done['checkpoint_sha256']!=seal['checkpoint_sha256']or done['forecast_targets_read']:raise ValueError('Forecast checkpoint mismatch or target access')
            if 'trajectory_sha256'in done and sha(predpath)!=done['trajectory_sha256']:raise ValueError('Forecast trajectory changed')
            reference=list((base/'pilot-01').glob(name+'-*/trajectory.npz'))
            if len(reference)!=1 or sha(reference[0])!=case['source_reference_sha256']:raise ValueError('Reference mismatch')
            with np.load(reference[0])as ref,np.load(predpath)as pred:
                actual=ref['surface_positions_m'];x=pred['surface_positions_m'];times=ref['times_s']
                if x.shape!=actual.shape or not np.allclose(pred['times_s'],times,atol=1e-9):raise ValueError('Complete matching trajectory required')
                if not np.isfinite(x).all():raise ValueError('Completed forecast has nonfinite coordinates')
                future=times>case['calibration_duration_s']+1e-10
                row.update(state='complete',future=metrics(actual[future],x[future]),forecast_frames=int(future.sum()),prediction_sha256=sha(predpath),reference_sha256=case['source_reference_sha256'])
    report=dict(scope='Development temporal forecasting; known boundary; coarse unqualified reference; unequal fitting budgets; no held OOD or superiority claim',planned_slots=len(rows),counts={method:{state:sum(r['method']==method and r['state']==state for r in rows)for state in sorted(set(r['state']for r in rows))}for method in labels},rows=rows)
    write(ROOT/'reports/complex-scenes-v18'/output,report);print(json.dumps(report['counts'],indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='fitted-development-progress.json');p.add_argument('--material-label',default='material-scene-fit-01');args=p.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Report basename required')
    audit(args.output,args.material_label)
