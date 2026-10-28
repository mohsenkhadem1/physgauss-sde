"""Evaluate the frozen v06 RGB-D initial uncertainty on fresh v07 test objects."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin import sequential_policy as old
from scripts.phystwin.v07_study import BASE,SOURCE,write
seal=json.loads((SOURCE/'analysis-rgbd/seal.json').read_text());old.verify(seal)
interp=old.interpolators();bias,C,_=old.fit(interp);ex=old.Experiment(interp,65,bias,C,.0004,'cuda:0');z=np.load(BASE/'test-observations.npz');pred=[]
for start in range(0,len(z['measurements']),4):pred.append(ex.model([]).infer(z['measurements'][start:start+4,:9]))
np.savez_compressed(BASE/'test-legacy-initial.npz',mean=np.concatenate([p['mean'] for p in pred]),sd=np.concatenate([p['sd'] for p in pred]),truth=z['truth'],scale=seal['scale'])
old.verify(seal)
