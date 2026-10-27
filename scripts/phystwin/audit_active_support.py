"""Post-run numerical audits; never retunes the frozen policy or test choices."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.active_support_reference import BASE,ROOT,protocol
from scripts.phystwin.active_support_policy import interpolators,fit_discrepancy,Experiment,data


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
    spec=protocol();interp=interpolators();mu,C,_=fit_discrepancy(spec,interp)
    ys,_=data(spec,'test');saved=np.load(BASE/'analysis/test-predictions.npz')['adaptive']
    ex=Experiment(interp,65,mu,C,spec['noise_sigma_m'],a.device);choices=[]
    for i,y0 in enumerate(ys[0]):
        choices.append(int(np.argmin(ex.choose(y0,1024)))+1)
        if i%16==0:print('AUDIT',i,len(ys[0]),flush=True)
    out=dict(coarse_grid=65,fine_grid=129,samples=1024,choice_agreement=int(np.sum(np.array(choices)==saved)),
        trials=len(saved),coarse_choices=choices,scope='Post-run diagnostic; published 129-grid policy choices are unchanged')
    (BASE/'analysis/test-choice-refinement.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
