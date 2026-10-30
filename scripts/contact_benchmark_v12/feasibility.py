"""Separate post-hoc local-volume guard; never changes the frozen v12 method.

Uses exact inherited FEM tetrahedra. This is a geometry repair, not an elastic
solve, collision-free projection, or continuous-time feasibility guarantee.
"""
import numpy as np
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.data import rigid_map


def inherited_elements(data,nonconvex=None):
    if 'evaluation' in data:raise ValueError('Evaluator truth is not a guard input')
    if nonconvex is None:nonconvex=data.get('dataset')=='PhysTwin'
    return ExternalFEM(data,device='cpu',nonconvex=nonconvex).elements.cpu().numpy().copy()


def jacobians(reference,points,elements):
    """Signed deformed/rest tetra volume ratio, same as det(F) in source FEM."""
    reference=np.asarray(reference,float);points=np.asarray(points,float);e=np.asarray(elements,int)
    Dm=np.stack([reference[e[:,k]]-reference[e[:,0]] for k in (1,2,3)],axis=-1)
    Ds=np.stack([points[e[:,k]]-points[e[:,0]] for k in (1,2,3)],axis=-1)
    denominator=np.linalg.det(Dm)
    if np.any(denominator==0) or not np.isfinite(denominator).all():raise ValueError('Invalid inherited rest elements; never remesh')
    return np.linalg.det(Ds)/denominator


def _temporal_summary(displacement,times):
    dt=np.diff(times);velocity=np.diff(displacement,axis=0)/dt[:,None,None]
    acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None,None]
    result={}
    for name,values in [('guard_displacement_m',displacement),('guard_velocity_change_m_s',velocity),('guard_acceleration_change_m_s2',acceleration)]:
        valid=np.isfinite(values).all(axis=(1,2))
        result[name]=dict(valid_frames=int(valid.sum()),total_frames=len(values),
                          max_coordinate_rms=float(np.sqrt(np.mean(values[valid]**2,axis=(1,2))).max()) if valid.any() else None)
    return result


def repair(data,base_prediction,candidate_prediction,nonconvex=None,elements=None):
    """Return (repaired nodes, audit metadata), without observing target motion.

    Call BEFORE contact rollout and velocity derivation. Per frame, rigidly align
    the inherited FEM shape to the proposed shape, then backtrack the residual
    using fixed alpha=1,1/2,...,2^-20,0. Choose the largest TESTED feasible alpha.
    Every inherited element must satisfy J > 0.03+1e-8, a numerical margin over
    the original FEM's strict 0.03 cutoff. An infeasible/nonfinite anchor or
    nonfinite proposal remains an explicit NaN failure, never silently repaired
    using another frame or observed target. No new topology is constructed.
    """
    if 'evaluation' in data:raise ValueError('Evaluator truth is not a guard input')
    base=np.asarray(base_prediction,float);candidate=np.asarray(candidate_prediction,float)
    if base.shape!=candidate.shape or base.ndim!=3 or base.shape[1:]!=np.asarray(data['nodes']).shape:raise ValueError('Mismatched nodal trajectories')
    reference=np.asarray(data['nodes']);elements=inherited_elements(data,nonconvex) if elements is None else np.asarray(elements,int)
    cutoff=.03+1e-8;schedule=[2.**(-k) for k in range(21)]+[0.]
    out=candidate.copy();rows=[]
    for t,(original,proposed) in enumerate(zip(base,candidate)):
        row=dict(frame=t)
        if not np.isfinite(original).all() or not np.isfinite(proposed).all():
            out[t]=np.nan;row.update(failed=True,reason='Nonfinite inherited anchor or proposal',alpha=None);rows.append(row);continue
        baseJ=jacobians(reference,original,elements)
        if not (baseJ>cutoff).all():
            out[t]=np.nan;row.update(failed=True,reason='Inherited FEM anchor violates volume cutoff',alpha=None,min_base_J=float(baseJ.min()));rows.append(row);continue
        proposedJ=jacobians(reference,proposed,elements)
        if (proposedJ>cutoff).all():
            row.update(failed=False,alpha=1.,min_base_J=float(baseJ.min()),min_proposal_J=float(proposedJ.min()),min_repaired_J=float(proposedJ.min()),guard_coordinate_rms_m=0.);rows.append(row);continue
        R,b=rigid_map(original,proposed);anchor=original@R+b
        anchorJ=jacobians(reference,anchor,elements)
        if not (anchorJ>cutoff).all():
            out[t]=np.nan;row.update(failed=True,reason='Rigid alignment precision violates anchor cutoff',alpha=None,min_base_J=float(baseJ.min()));rows.append(row);continue
        residual=proposed-anchor
        for alpha in schedule[1:]:
            trial=anchor+alpha*residual;J=jacobians(reference,trial,elements)
            if (J>cutoff).all():break
        else:
            out[t]=np.nan;row.update(failed=True,reason='No feasible tested residual scale including zero',alpha=None);rows.append(row);continue
        out[t]=trial
        row.update(failed=False,alpha=float(alpha),min_base_J=float(baseJ.min()),min_proposal_J=float(proposedJ.min()),min_repaired_J=float(J.min()),
                   proposed_residual_coordinate_rms_m=float(np.sqrt(np.mean(residual**2))),
                   guard_coordinate_rms_m=float(np.sqrt(np.mean((trial-proposed)**2))))
        rows.append(row)
    repaired=[r for r in rows if not r['failed'] and r['alpha']<1.]
    meta=dict(method='inherited-FEM local-volume feasibility guard',cutoff_J=.03,numerical_margin=1e-8,
              alpha_schedule=schedule,element_count=len(elements),frames=rows,
              failed_frames=[r['frame'] for r in rows if r['failed']],modified_frames=[r['frame'] for r in repaired],
              modified_fraction=len(repaired)/len(rows) if rows else 0.,
              placement='before contact rollout and derived velocities',
              temporal_change=_temporal_summary(out-candidate,np.asarray(data['times'])[:len(base)]),
              limitations=['Local orientation only; no global selfcollision certification','Not an equilibrium solve','Backtracking may introduce temporal velocity/acceleration changes','Reference skinny tetrahedra retained, never remeshed'])
    return out,meta
