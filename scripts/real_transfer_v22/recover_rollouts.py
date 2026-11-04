"""Retry numerical rollout failures without refitting or relaxing tolerance."""
import json,shutil,time
import numpy as np,torch
from .campaign import ROOT,LABEL,CASES,write,sha
from scripts.real_transfer_v19.phystwin_run import load
from scripts.real_transfer_v19.neural_fem import NeuralFEM
from scripts.external_benchmark.fem import ExternalFEM
from pathlib import Path

def main():
    torch.set_num_threads(1);recoveries=[]
    for case in CASES:
        for method in ('neural','analytic'):
            folder=ROOT/'runs/real-transfer-v19/fits'/LABEL/case/method
            meta=json.loads((folder/'complete.json').read_text())
            if not meta['failed_frames']:continue
            data,_=load(case,240,False)
            model=NeuralFEM(data,'cuda:0') if method=='neural' else ExternalFEM(data,'cuda:0',field=False,nonconvex=True,materials=6)
            with np.load(folder/'prediction.npz') as f:theta=f['theta'].copy();nodes=f['nodes'].copy()
            params=model.t(theta);x=None;pred=[];diagnostics=[];retry=[]
            for frame in range(len(data['times'])):
                try:x,_,d=model.solve(params,frame,x)
                except (RuntimeError,ValueError) as e:
                    x,_,d=model.solve(params,frame,x,maxiter=320)
                    retry.append(dict(frame=frame,original_reason=str(e),retry_maxiter=320,diagnostic=d))
                pred.append(x.detach().cpu().numpy()*model.scale+model.origin);diagnostics.append(d)
            assert np.isfinite(pred).all()
            backup=folder/'pre-recovery';backup.mkdir(exist_ok=False)
            for name in ('prediction.npz','complete.json'):shutil.copy2(folder/name,backup/name)
            np.savez_compressed(folder/'prediction.npz',predicted=np.asarray(pred),nodes=nodes,theta=theta)
            recovery=dict(policy='Same warm-start trajectory and tolerance; on failure retry with 320 instead of 80 iterations. No refit.',
                retries=retry,original_prediction_sha256=meta['prediction_sha256'],script_sha256=sha(Path(__file__)),script=str(Path(__file__).relative_to(ROOT)))
            meta.update(failed_frames=[],diagnostics=diagnostics,prediction_sha256=sha(folder/'prediction.npz'),recovery=recovery)
            write(folder/'complete.json',meta);recoveries.append(dict(case=case,method=method,**recovery));print(case,method,retry,flush=True)
    write(ROOT/'reports/real-transfer-v22/solver-recovery.json',dict(recoveries=recoveries))

if __name__=='__main__':main()
