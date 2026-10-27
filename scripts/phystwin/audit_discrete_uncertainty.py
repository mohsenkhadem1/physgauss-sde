"""Replay v01 observations; isolate finite-support collapse from mesh discrepancy."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.controlled_probing import normalize, SIGMA

ROOT=Path(__file__).resolve().parents[2]


def main():
    base=ROOT/'runs/phystwin-transfer-v01';out=ROOT/'reports/phystwin-uncertainty-v02'
    bank=np.load(base/'probing/bank.npz');fields=bank['fields'];states=bank['states']
    rest=np.load(base/'reference/soft_firm-axial-n2.npz')['rest']
    end=np.isclose(rest[:,0],.24);marker=np.argmin(np.linalg.norm(rest-[.12,.04,.04],axis=1))
    prefix=fields[:,0,end].reshape(9,-1);probe=fields[:,0,marker];target=fields[:,2].reshape(9,-1)
    order=np.argsort(target,axis=0);values=np.take_along_axis(target,order,axis=0)
    observations=json.loads((base/'probing/observations.json').read_text())
    records=[]
    for row in observations:
        p=normalize(-.5*np.sum(((prefix-row['prefix'])/SIGMA)**2,axis=1))
        post=normalize(np.log(np.maximum(p,1e-300))-.5*np.sum(((probe-np.array(row['reset_probes'])[0])/SIGMA)**2,axis=1))
        physical=[2000,1000] if row['object']=='soft_firm' else [8000,64]
        state=int(np.flatnonzero(np.all(states==physical,axis=1))[0])
        fine=(np.load(base/'reference'/f"{row['object']}-oblique-n8.npz")['sampled_positions'][-1]-rest).ravel()
        for phase,weights in [('before',p),('after',post)]:
            cumulative=np.cumsum(weights[order],axis=0)
            lo=values[np.argmax(cumulative>=.05,axis=0),np.arange(243)]
            hi=values[np.argmax(cumulative>=.95,axis=0),np.arange(243)]
            records.append(dict(phase=phase,collapsed_fraction=float(np.mean(hi-lo<1e-12)),
                coarse_truth_coverage=float(np.mean((target[state]>=lo-1e-12)&(target[state]<=hi+1e-12))),
                fine_truth_coverage=float(np.mean((fine>=lo-1e-12)&(fine<=hi+1e-12)))))
    summary={phase:{k:float(np.mean([r[k] for r in records if r['phase']==phase]))
                     for k in ['collapsed_fraction','coarse_truth_coverage','fine_truth_coverage']}
             for phase in ['before','after']}
    summary['scope']='Same saved noisy observations and posterior; only scoring truth changes from exact n4 bank state to independent n8 field. Diagnostic, not new test.'
    out.mkdir(exist_ok=True);(out/'discrete-ablation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
