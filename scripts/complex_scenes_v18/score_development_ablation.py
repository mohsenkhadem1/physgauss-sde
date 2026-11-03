"""Score already sealed predictions; never called by calibration optimizers."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.audit_development import shape_rms
from scripts.complex_scenes_v18.run_development import ROOT,write


def main():
    case='clamp_release-v0';base=ROOT/'runs/complex-scenes-v18';meta=json.loads((base/'development-prefixes'/case/'metadata.json').read_text())
    source=next((base/'pilot-01').glob(case+'-*/trajectory.npz'))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['source_reference_sha256']:raise ValueError('Reference changed')
    truth=np.load(source);actual=truth['surface_positions_m'];times=truth['times_s'];future=times>meta['calibration_duration_s']+1e-10;rows=[]
    for mode in ['frozen','bulk-extension']:
        folder=base/'material-scale-fit-01'/mode/case;seal=json.loads((folder/'fit-seal.json').read_text());raw=folder/'sealed-rollout'/case
        done=json.loads((raw/'complete.json').read_text());path=raw/'trajectory.npz'
        if not seal['completed']or not done['completed']or seal['forecast_targets_read']or done['parameters']!=seal['best']['parameters']:raise ValueError('Unsealed/incomplete prediction')
        prediction=np.load(path);x=prediction['surface_positions_m']
        if x.shape!=actual.shape or not np.allclose(prediction['times_s'],times,atol=1e-9):raise ValueError('Complete matching trajectory required')
        error=x[future]-actual[future];shape=np.array([[shape_rms(a,p)for a,p in zip(aa,pp)]for aa,pp in zip(actual[future],x[future])])
        rows.append(dict(method=mode,calibration_visible_vector_rms_mm=1000*np.sqrt(seal['best']['loss_m2']),future_vector_rms_mm=1000*np.sqrt(np.mean(np.sum(error**2,axis=-1))),future_rigid_aligned_shape_rms_mm=1000*np.sqrt(np.mean(shape**2)),parameters=seal['best']['parameters'],calibration_evaluations=len(seal['trials']),optimizer_converged=seal['optimizer_converged'],fit_seal_sha256=hashlib.sha256((folder/'fit-seal.json').read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(scope='One development clamp-release case; known boundary; coarse unqualified reference; unequal fitting budgets; no held-world superiority claim',case=case,calibration_duration_s=meta['calibration_duration_s'],forecast_frames=int(future.sum()),reference_sha256=meta['source_reference_sha256'],rows=rows)
    write(ROOT/'reports/complex-scenes-v18/material-development-ablation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
