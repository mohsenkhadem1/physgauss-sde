"""Prefix-only numerical initialization repair; original frozen objective unchanged.

Only used after a job reports no valid initial fitting iterate. Preserve that
original failure in the campaign log. Select the first feasible start, not the
best held-out result. No future trajectory or score enters this decision.
"""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import scipy.optimize
original=scipy.optimize.minimize

def feasible_start(fun,x0,*args,**kwargs):
    bounds=kwargs['bounds'];G=(len(x0)-5) # SoftVTBench: 1 material, field, floor, contact, gravity.
    if G!=2:raise ValueError('This repair is restricted to the two-grip SoftVTBench configuration')
    candidates=[np.asarray(x0).copy()]
    for gravity,attachment in [(-6.,-1.),(-6.,-3.),(-8.,-3.)]:
        candidate=np.asarray(x0).copy();candidate[0]=gravity;candidate[1:1+G]=attachment;candidates.append(candidate)
    for index,candidate in enumerate(candidates):
        value,_=fun(candidate)
        print('FEASIBILITY_START',index,candidate.tolist(),value,flush=True)
        if np.isfinite(value) and value<1000:
            return original(fun,candidate,*args,**kwargs)
    raise RuntimeError('All declared prefix-only initial guesses failed')
scipy.optimize.minimize=feasible_start
runpy.run_module('scripts.external_benchmark.fit',run_name='__main__')
