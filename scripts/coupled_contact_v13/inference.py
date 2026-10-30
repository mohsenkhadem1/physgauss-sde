"""Calibration-only discrete physical hypotheses. No future target argument."""
import numpy as np
from scripts.coupled_contact_v13.baselines import fit_baselines

def observation_nodes(rest,n=18):
    low=rest.min(0);high=rest.max(0)
    boundary=np.flatnonzero(np.any(np.isclose(rest,low)|np.isclose(rest,high),axis=1))
    return boundary[np.linspace(0,len(boundary)-1,n).round().astype(int)]

def choose(observed,mask,candidates):
    observed=np.asarray(observed);mask=np.asarray(mask,bool)
    scores=[]
    if not mask.any():return dict(success=False,reason='No observed positions',scores_m2=[float('inf')]*len(candidates))
    for row in candidates:
        pred=row.get('frames')
        valid=pred is not None and pred.shape==observed.shape and np.isfinite(pred).all()
        score=float(np.mean((pred[mask]-observed[mask])**2)) if valid else float('inf')
        scores.append(score)
    finite=np.isfinite(scores)
    if not finite.any():return dict(success=False,reason='All calibration candidates failed',scores_m2=scores)
    index=int(np.argmin(scores));row=candidates[index];order=np.sort(np.asarray(scores)[finite])
    return dict(success=True,index=index,scale=row['scale'],friction=row['friction'],calibration_rmse_m=float(np.sqrt(scores[index])),scores_m2=scores,
                second_best_gap_m2=float(order[1]-order[0]) if len(order)>1 else None,
                scale_at_grid_edge=row['scale'] in (min(c['scale'] for c in candidates),max(c['scale'] for c in candidates)),
                friction_at_grid_edge=row['friction'] in (min(c['friction'] for c in candidates),max(c['friction'] for c in candidates)))

def calibrate(rest,observed,mask,tools,times,active,libraries):
    selected={name:choose(observed,mask,candidates) for name,candidates in libraries.items()}
    empirical=fit_baselines(rest,observed,tools,times,mask,active,ridge=.1)
    return selected,empirical
