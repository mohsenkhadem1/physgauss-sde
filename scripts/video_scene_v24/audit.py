"""Post-run provenance and arithmetic checks; does not tune the model."""
import json, subprocess
import numpy as np
from .protocol import ROOT,BASE,REPORT,sha,write

def main():
    plan=json.loads((REPORT/'protocol.json').read_text()); results=json.loads((REPORT/'results.json').read_text())
    groups=list(plan['cases'].values()); assert len(set(sum(groups,[])))==sum(map(len,groups))==22
    training={}
    for method in plan['methods']:
        folder=BASE/method; p=json.loads((folder/'plan.json').read_text()); c=json.loads((folder/'complete.json').read_text())
        assert p['train_cases']==plan['cases']['train']
        assert p['protocol_sha256']==sha(REPORT/'protocol.json')
        assert sha(folder/'model.pt')==c['checkpoint_sha256']==results['checkpoint_sha256'][method]
        for f,h in c['source_sha256'].items(): assert sha(ROOT/f)==h
        for case,h in p['input_manifests'].items(): assert sha(BASE/'prepared'/case/'complete.json')==h
        training[method]=dict(plan=p,completion=c)
    for row in results['rows']:
        path=BASE/'predictions'/row['case']/'predictions.npz'; assert sha(path)==row['prediction_sha256']
        with np.load(path) as data:
            for arr in data.values(): assert np.isfinite(arr).all()
    for group,methods in results['summary'].items():
        for method,metrics in methods.items():
            for metric,value in metrics.items():
                recomputed=np.mean([r['results'][method][metric] for r in results['rows'] if r['partition']==group])
                assert abs(recomputed-value)<1e-10
    videos=json.loads((REPORT/'videos.json').read_text())['videos']
    for v in videos:
        subprocess.run([str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'),'-v','error','-i',str(ROOT/v['movie']),'-f','null','-'],check=True)
    write(REPORT/'training.json',training)
    write(REPORT/'audit.json',dict(passed=True,cases=22,videos_decoded=len(videos),checks=['disjoint case groups','source/checkpoint/protocol/input-manifest hashes','finite predictions','aggregate arithmetic','full video decode'],limitations='Same-agent audit; not independent scientific validation. Labels are author reconstructed tracks.'))
    print('Audit passed: 22 cases, 3 videos, provenance and aggregate arithmetic.')
if __name__=='__main__': main()
