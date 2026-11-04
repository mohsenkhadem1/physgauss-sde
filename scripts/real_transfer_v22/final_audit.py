"""Verify sealed predictions, all-case completion and unrecorded camera poses."""
import json,hashlib,pickle,subprocess
import numpy as np
from .campaign import ROOT,LABEL,CASES,write

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=ROOT/'reports/real-transfer-v22';checks=[]
    protocol=json.loads((report/'protocol.json').read_text())
    for path,digest in protocol['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
        checks.append('source '+path)
    for case in CASES:
        for method in ('neural','analytic'):
            folder=ROOT/'runs/real-transfer-v19/fits'/LABEL/case/method
            meta=json.loads((folder/'complete.json').read_text())
            assert meta['completed'] and not meta['failed_frames'],(case,method)
            if 'recovery' in meta:
                assert sha(ROOT/meta['recovery']['script'])==meta['recovery']['script_sha256']
                assert sha(folder/'pre-recovery/prediction.npz')==meta['recovery']['original_prediction_sha256']
            assert all(d['minJ']>0 for d in meta['diagnostics'])
            for key,file in [('prediction_sha256','prediction.npz'),('fit_sha256','fit.json'),('plan_sha256','plan.json')]:assert sha(folder/file)==meta[key]
            with np.load(folder/'prediction.npz') as f:assert np.isfinite(f['predicted']).all()
            plan=json.loads((folder/'plan.json').read_text());assert plan['input']['mask_version']=='incoming-v22'
            checks.append('sealed finite corrected fit '+case+' '+method)
    for key in CASES+['single_lift_sloth--single_push_sloth','single_lift_zebra--double_lift_zebra']:
        out=ROOT/'runs/real-transfer-v22/novel-views'/key
        m=json.loads((out/'manifest.json').read_text());case=m['case'];assert m['frames']==120
        for path,digest in m['input_sha256'].items():assert sha(ROOT/path)==digest,path
        with (ROOT/'runs/third_party/PhysTwin/data/different_types'/case/'calibrate.pkl').open('rb') as f:C=np.asarray(pickle.load(f))
        novel=np.asarray(m['camera_to_world']);assert np.isfinite(novel).all()
        assert all(not np.allclose(n,c) for n in novel for c in C)
        for pose in novel:
            np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
            assert np.linalg.det(pose[:3,:3])>.999
        video=ROOT/m['movie'];assert video.stat().st_size>10000
        # Decode the entire movie, failing on corrupt packets.
        subprocess.run([str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'),'-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        checks.append('novel cameras and decoded video '+key)
    write(report/'verification.json',dict(passed=True,checks=checks,total=len(checks),
        note='Unrecorded camera poses verified; no novel-view RGB accuracy claim. Adapter masks audited, upstream tracking not certified causal.'))
    print(len(checks),'checks passed')

if __name__=='__main__':main()
