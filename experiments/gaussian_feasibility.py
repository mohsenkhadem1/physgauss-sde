"""Explicit feasibility conditioning of learned physical-posterior samples.

The condition concerns a neural trajectory on a finite query grid. It performs
no FEM solve, does not repair particles, and does not establish continuous-time
or continuum feasibility. Its normalizing probability is estimated by sampling.
"""
import numpy as np
import torch


def volume_feasibility(positions,reference,minimum_ratio=.1):
    """Return one pass/fail decision and minimum signed tet ratio per particle."""
    q=torch.as_tensor(positions)
    if q.ndim!=4:raise ValueError('Expected [particles,times,nodes,3]')
    rest=torch.as_tensor(reference['rest_points_m'],device=q.device,dtype=q.dtype)
    tets=torch.as_tensor(reference['tetrahedra'],device=q.device,dtype=torch.long)
    initial=torch.linalg.det(rest[tets[:,1:]]-rest[tets[:,:1]])
    if not torch.isfinite(initial).all() or not (initial>0).all():raise ValueError('Reference tetrahedra must be positive')
    finite=torch.isfinite(q).all((1,2,3))
    ratios=torch.linalg.det(q[:,:,tets[:,1:]]-q[:,:,tets[:,:1]])/initial
    minimum=ratios.flatten(1).amin(1)
    valid=finite&torch.isfinite(minimum)&(minimum>=minimum_ratio)
    return {'valid':valid,'finite':finite,'minimum_volume_ratio':minimum}


def physical_range_audit(particles,valid=None):
    """Compare draws with declared simulation-design ranges, without truth labels."""
    p=particles['effective_physical'].detach().cpu().numpy()
    bc=particles['boundary_type'].detach().cpu().numpy()
    force=np.linalg.norm(p[:,16:19],axis=-1)
    lower=np.where(bc==0,.03,.12);upper=np.where(bc==0,.06,.24)
    material=(p[:,0]>=np.log(1500))&(p[:,0]<=np.log(20000))
    ranges=(force>=lower)&(force<=upper)
    moving=bc==3;motion=np.linalg.norm(p[:,22:25],axis=-1)
    result={'particles':len(p),'force_norm_N_quantiles':np.quantile(force,[0,.05,.5,.95,1]).tolist(),
            'force_below_design_fraction':float((force<lower).mean()),
            'force_above_design_fraction':float((force>upper).mean()),
            'material_E_outside_design_fraction':float((~material).mean()),
            'moving_particles':int(moving.sum()),
            'moving_amplitude_m_quantiles':np.quantile(motion[moving],[0,.05,.5,.95,1]).tolist() if moving.any() else None,
            'moving_amplitude_outside_8mm_plusminus_half_mm_fraction':float((abs(motion[moving]-.008)>.0005).mean()) if moving.any() else None,
            'design_ranges':'Force norm .03-.06N for point pin, .12-.24N otherwise; E1500-20000Pa; moving amplitude .008m',
            'qualification':'Diagnostics only; these range conditions are not rejection criteria'}
    if valid is not None:
        valid=np.asarray(valid,dtype=bool)
        result['invalid_with_force_inside_design_count']=int((~valid&ranges).sum())
        result['invalid_with_force_and_E_inside_design_count']=int((~valid&ranges&material).sum())
    return result


def feasible_posterior_samples(bundle,encoded,reference,times,samples=64,candidate_budget=256,
                               seed=20261102,minimum_ratio=.1,chunk_size=8):
    """Fixed-budget rejection with complete counts; never silently refill.

    Returns paired raw first-K and feasible first-K predictions. The latter
    approximate q(theta|D, neural J(query_grid)>=threshold). Their normalized
    density is not available. Invalid particles are counted and never repaired.
    """
    from evaluate_gaussian_inverse import posterior_samples,predict_particles
    if candidate_budget<samples or samples<2:raise ValueError('Budget must cover at least two requested samples')
    candidates=posterior_samples(bundle,encoded,candidate_budget,seed)
    flags=[];minimum=[];finite=[];raw=[];retained=[];retained_ids=[]
    for start in range(0,candidate_budget,chunk_size):
        stop=min(start+chunk_size,candidate_budget)
        block={k:v[start:stop] for k,v in candidates.items()}
        q=predict_particles(bundle,block,reference,times,chunk_size)
        audit=volume_feasibility(q,reference,minimum_ratio)
        flags.extend(audit['valid'].cpu().tolist());minimum.extend(audit['minimum_volume_ratio'].cpu().tolist())
        finite.extend(audit['finite'].cpu().tolist())
        if start<samples:raw.append(q[:min(samples-start,len(q))].cpu())
        for local in torch.nonzero(audit['valid'],as_tuple=False).flatten().tolist():
            if len(retained_ids)<samples:
                retained_ids.append(start+local);retained.append(q[local].cpu())
    accepted=np.asarray(flags,dtype=bool);mass=float(accepted.mean())
    ids=torch.tensor(retained_ids,device=candidates['physical'].device,dtype=torch.long)
    particles={k:v[ids] for k,v in candidates.items()}
    stats={'requested_samples':samples,'candidate_budget':candidate_budget,'valid_candidates':int(accepted.sum()),
           'rejected_candidates':int((~accepted).sum()),'nonfinite_candidates':int((~np.asarray(finite)).sum()),
           'retained_samples':len(retained_ids),'insufficient_support':len(retained_ids)<samples,
           'estimated_raw_retained_probability':mass,'retained_probability_mc_standard_error':float(np.sqrt(mass*(1-mass)/candidate_budget)),
           'minimum_required_volume_ratio':minimum_ratio,'minimum_candidate_volume_ratio':float(np.nanmin(minimum)),
           'query_times_s':np.asarray(times).tolist(),'seed':seed,
           'raw_range_audit':physical_range_audit(candidates,accepted),
           'retained_range_audit':physical_range_audit(particles) if retained_ids else None,
           'distribution':'Learned posterior conditioned on neural signed tet ratio >= threshold at the finite query grid',
           'qualification':'No particle repair, no hidden-state selection, no exact normalized density, no continuum or calibration claim'}
    return {'raw_predictions':torch.cat(raw).numpy(),
            'predictions':torch.stack(retained).numpy() if retained else None,
            'particles':particles,'stats':stats,'candidate_valid':accepted,'candidate_minimum_volume_ratio':np.asarray(minimum),
            'raw_particles':{k:v[:samples] for k,v in candidates.items()}}
