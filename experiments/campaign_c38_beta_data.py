"""Fresh object-wise simulation pairs for one declared analytical log-NH family.

No study draw occurs at import. Eight contexts share each object's physical
latents and counterfactual acquisition noise. Perception/dynamics are absent.
"""
from __future__ import annotations
import numpy as np
from experiments.campaign_c38_bar_reference import prior_physical,endpoint_displacements,oracle_positions,material_quadrature,TARGET_FORCE,SIGMA_M
from experiments.campaign_c38_beta_posterior import canonical_inputs

CONTEXTS=('prefix_left','prefix_right','left_then_left','left_then_right','left_then_poke_left','right_then_left','right_then_right','right_then_poke_right')


def contexts_from_latents(unit,prefix_noise,action_noise):
    u=np.asarray(unit,dtype=float);pn=np.asarray(prefix_noise,dtype=float);an=np.asarray(action_noise,dtype=float);theta=prior_physical(u);B=len(theta)
    if pn.shape!=(B,2) or an.shape!=(B,4) or not np.isfinite(pn).all() or not np.isfinite(an).all():raise ValueError('Two prefix and four acquisition-noise values per independent object required')
    ends=endpoint_displacements(theta,[1.,2.]);prefix=ends[:,0]+pn;acquired=np.concatenate([ends[:,0],ends[:,1]],axis=1)+an
    forces=np.zeros((B,8,2));sides=np.zeros((B,8,2));observed=np.full((B,8,2),np.nan);mask=np.zeros((B,8,2),dtype=bool)
    for context in range(8):
        side=0 if context in (0,2,3,4) else 1
        forces[:,context,0]=1;sides[:,context,0]=side;observed[:,context,0]=prefix[:,side];mask[:,context,0]=True
        if context>=2:
            acquisition={2:0,3:1,4:2,5:0,6:1,7:3}[context]
            forces[:,context,1]=1 if acquisition<2 else 2;sides[:,context,1]=acquisition%2;observed[:,context,1]=acquired[:,acquisition];mask[:,context,1]=True
    features=canonical_inputs(forces.reshape(-1,2),sides.reshape(-1,2),observed.reshape(-1,2),mask.reshape(-1,2)).reshape(B,8,10)
    return dict(features=features,forces=forces,sides=sides,observed_m=observed,mask=mask),dict(unit=u.copy(),theta=theta,canonical_positions_m=oracle_positions(theta,TARGET_FORCE,material_quadrature()))


def simulate_objects(count,seed):
    if type(count) is not int or count<1 or type(seed) is not int:raise ValueError('Prospectively fixed object count and seed required')
    # Independent streams; no conditional redraw/drop based on outcomes.
    streams=np.random.SeedSequence(seed).spawn(4);u=np.random.default_rng(streams[0]).random((count,2))
    if ((u<=0)|(u>=1)).any():raise FloatingPointError('Noninterior prior draw; preserve failure, no redraw')
    pn=np.random.default_rng(streams[1]).normal(0,SIGMA_M,(count,2));an=np.random.default_rng(streams[2]).normal(0,SIGMA_M,(count,4));camera=np.random.default_rng(streams[3]).integers(0,2,count)
    public,private=contexts_from_latents(u,pn,an);prefix=dict(features=public['features'][np.arange(count),camera],camera=camera)
    return public,private,prefix
