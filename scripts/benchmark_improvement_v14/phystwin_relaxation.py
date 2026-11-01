"""Causal single-time-constant relaxation of nonrigid equilibrium response."""
import numpy as np
from scripts.external_benchmark.data import rigid_map
from scripts.contact_benchmark_v12.object_frame import _observed_prediction
TAU_GRID=(0.,.033,.067,.1,.2,.4)

def filtered(data,base,tau):
    base=np.asarray(base,float)
    if tau==0:return base.copy()
    nodes=np.asarray(data['nodes']);out=base.copy();state=None
    # Material frame follows inherited rigid pose; only deformation is delayed.
    for t,points in enumerate(base):
        R,b=rigid_map(nodes,points);equilibrium=(points-b)@R.T-nodes
        if t==0:state=equilibrium.copy()
        else:
            dt=float(data['times'][t]-data['times'][t-1]);decay=np.exp(-dt/tau)
            state=decay*state+(1-decay)*equilibrium
        out[t]=(nodes+state)@R+b
    out[0]=base[0]
    return out

def predict(data,base_prediction=None,config=None):
    if 'evaluation'in data:raise ValueError('Evaluator forbidden')
    end=int(data['train_end']);base=np.asarray(base_prediction,float)
    if len(data['observed'])!=end or not np.isfinite(base).all():raise ValueError('Strict prefix and finite anchor required')
    valid=np.asarray(data['observed_mask'],bool)&np.isfinite(data['observed']).all(2)
    if not valid.any():raise ValueError('No valid prefix')
    scores=[];predictions=[]
    for tau in TAU_GRID:
        p=filtered(data,base,tau);projected=_observed_prediction(data,p[:end]);scores.append(float(np.mean((projected[valid]-data['observed'][valid])**2)));predictions.append(p)
    index=int(np.argmin(scores))
    return predictions[index],dict(method='causal_material_deformation_relaxation',tau_seconds=TAU_GRID[index],grid_seconds=list(TAU_GRID),prefix_coordinate_rmse_mm=(np.sqrt(scores)*1000).tolist(),fit_end=end,initial_preserved=True,scope='Empirical viscoelastic relaxation of nonrigid equilibrium only; inherited rigid motion unchanged; no force/material identification')
