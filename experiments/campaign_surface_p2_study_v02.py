"""Prospective v02 ideal-static P2 study adapter and bounded policy statistics.

No automatic simulations, inference, cohort generation or I/O. The v01 .24N
catalog is preserved. This unfrozen study draft delegates the prospective
normal .08N / axial .18N-per-face v03 external force table. No study data
or posterior has been generated under either draft load budget.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from experiments.campaign_surface_p2_study_design_v01 import (
    ACTION_CENTERS,TARGET_CENTERS,CATALOG_CENTERS,observation_points,
    COHORT_SEED,PREFIX_NOISE_SEED,PROBE_NOISE_SEED,RANDOM_ACTION_SEED)

BOOTSTRAP_SEED=2026102485
OBJECTS=16
PREFIX='normal_x180'
TARGETS=tuple(f'{family}_x{round(c*1000):03d}' for family in ('normal','paired_axial') for c in TARGET_CENTERS)
ACTIONS={family:tuple(f'{family}_x{round(c*1000):03d}' for c in ACTION_CENTERS) for family in ('normal','paired_axial')}
SIGMA_M=20e-6
NORMAL_FORCE_N=.08
AXIAL_FORCE_N_PER_FACE=.18


def load_catalog(n):
    from experiments.campaign_surface_p2_force_v03 import load_catalog as shared_catalog,NORMAL_FORCE_N as normal,AXIAL_FORCE_N_PER_FACE as axial
    if normal!=NORMAL_FORCE_N or axial!=AXIAL_FORCE_N_PER_FACE:raise ValueError('Prospective shared-force budget changed')
    return shared_catalog(n)


def design():
    return dict(schema='surface-p2-study-v02',status='PROSPECTIVE_NO_OUTCOMES',target_mesh_n=4,
        unknowns=['E','K'],prior='Independent uniform logE[log2000,log8000],logK[log5,log30]',
        observation='24 ideal tracked 3D displacement channels; independent known Gaussian20um per channel',
        scope='Static finite-model diagnostic. No RGB, camera, dynamics, dropping or real-material calibration claim',
        objects=OBJECTS,cohort_seed=COHORT_SEED,prefix_noise_seed=PREFIX_NOISE_SEED,probe_noise_seed=PROBE_NOISE_SEED,
        random_action_seed=RANDOM_ACTION_SEED,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_clusters=16,bootstrap_repetitions=10000,
        prefix=PREFIX,target_labels=list(TARGETS),actions={k:list(v)for k,v in ACTIONS.items()},
        actions_targets_disjoint=True,normal_force_N=NORMAL_FORCE_N,paired_axial_force_N_per_face=AXIAL_FORCE_N_PER_FACE,
        external_force_version='campaign_surface_p2_force_v03',
        development_tuning=dict(before_any_actual_study=True,old_normal_force_N=.10,new_normal_force_N=.08,
            axial_force_per_face_unchanged_N=.18,trigger='Central v02 case00 NH conditional medium spatial indicator21.24um exceeded20um gate',
            rationale='Prospective20percent normal-force reduction; preserve prior normal.10 failure; no guarantee of nonlinear pass and no gate change'),
        post_probe_noise_repetitions=3,noise_pairing='One24-channel prefix and three independent24-channel reset noise vectors per object; same vectors across actions and data laws; objects independent',
        data_laws=['matching_linear','NH80_stress'],all16_retained=True,no_redraws=True,
        methods=['prior_predictive_mean','highest_likelihood_quadrature_node_plugin','posterior_predictive_mean','known_true_parameter_linear_oracle'],
        policies=['covariance_best_linear_target_MSE_reduction','each_of_three_fixed_actions','uniform_random_exact_three_action_average','Gaussian_observation_information'],
        families_separate=True,force_budget_scope='Matched force budget within each three-action family only; normal net.08N and paired axial net.36N are not matched force or energy across families; no pooled policy advantage',random_realization='Seeded ancillary draw recorded locally; primary random baseline is exact action average',
        numerical='Explicit protocol choice: preserved tensor baseline or conditional-amplitude positive irregular hypotheses; joint likelihood uses prefix+independent reset probe data; fail closed',
        numerical_options=dict(tensor='Simpson129/257/513/1025 +Gauss128/256/512 +six partialCDF checks',conditional_amplitude='Separate inner32/64/128 andouter128/256/512 refinements plus independent original-a adaptive evidence+sixCDF at eps1e-6/1e-8; reportederrors<=1e-5, warnings unresolved'),
        physics='All strain/J/equilibrium/stability gates on all loads; prefix amplitude>=100um and nonrigid fraction>=.3; other amplitudes descriptive. Full catalog spatial/quadrature admission required before execution',
        metrics='Latent disjoint96-coordinate target RMS;3noise repetitions averaged within each of16 objects; paired object bootstrap10000 seed2026102485',
        comparison='Every fixed policy separately, uniform-random exact average and information baseline; no advantage claim unless all16 physics/numerical valid',
        private='Unit coords, E/K, noise, observations, CDF/quantiles, probabilities, full fields and models stay LOCAL; publish aggregate scalar metrics/source hashes only')


def streamed_utilities(unit,weights,predict,action_labels,target_labels,*,chunk=512):
    """Two-pass centered covariance; workspace scales with chunk, not grid size.

    predict(unit,labels) returns [batch,labels,channels]. Only prefix posterior
    inputs enter this pure function. No observations, truth or actual outcomes.
    """
    u=np.asarray(unit,float);w=np.asarray(weights,float);actions=tuple(action_labels);targets=tuple(target_labels)
    if u.ndim!=2 or u.shape[1]!=2 or w.shape!=(len(u),) or len(u)<2 or chunk<1 or not actions or not targets:
        raise ValueError('Common weighted unit grid, nonempty labels and positive chunk')
    if not np.isfinite(u).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,atol=1e-12,rtol=0):raise ValueError('Finite normalized nonnegative weights')
    labels=actions+targets;anchor=np.asarray(predict(u[:1],labels),float)[0]
    if anchor.ndim!=2 or anchor.shape[0]!=len(labels)or not np.isfinite(anchor).all():raise ValueError('Finite label/channel predictions')
    channels=anchor.shape[1];mean=np.zeros_like(anchor)
    for start in range(0,len(u),chunk):
        y=np.asarray(predict(u[start:start+chunk],labels),float)
        if y.shape!=(min(chunk,len(u)-start),*anchor.shape)or not np.isfinite(y).all():raise ValueError('Invalid prediction chunk')
        mean+=np.einsum('b,blq->lq',w[start:start+len(y)],y-anchor)
    aa=np.zeros((len(actions),channels,channels));ta=np.zeros((len(actions),len(targets)*channels,channels));tt=0.
    for start in range(0,len(u),chunk):
        y=np.asarray(predict(u[start:start+chunk],labels),float)-anchor-mean;p=w[start:start+len(y)]
        a=y[:,:len(actions)];t=y[:,len(actions):].reshape(len(y),-1)
        aa+=np.einsum('b,bai,baj->aij',p,a,a);ta+=np.einsum('b,bi,baj->aij',p,t,a)
        tt+=float(np.einsum('b,bi,bi->',p,t,t)/(len(targets)*channels))
    scores=[]
    for A,T in zip(aa,ta):
        factor=cho_factor(A+SIGMA_M**2*np.eye(channels),lower=True)
        reduction=float(np.sum(T*cho_solve(factor,T.T).T)/(len(targets)*channels))
        if reduction< -1e-20 or reduction>tt+max(1e-20,tt*1e-10):raise FloatingPointError('Invalid linear covariance risk reduction')
        sign,logdet=np.linalg.slogdet(np.eye(channels)+A/SIGMA_M**2)
        if sign<=0:raise FloatingPointError('Invalid observation covariance')
        scores.append(dict(expected_linear_target_MSE_reduction_m2=reduction,prior_target_MSE_m2=tt,
            expected_linear_remaining_target_MSE_m2=tt-reduction,Gaussian_observation_information_nats=float(.5*logdet)))
    return scores


def choices_from_prefix(unit,probability,predict,*,chunk=512):
    """Pure causal selection API. Deterministic first action wins exact ties."""
    result={}
    for family,labels in ACTIONS.items():
        scores=streamed_utilities(unit,probability,predict,labels,TARGETS,chunk=chunk)
        result[family]=dict(actions=list(labels),scores=scores,
            proposed=int(np.argmax([s['expected_linear_target_MSE_reduction_m2']for s in scores])),
            information=int(np.argmax([s['Gaussian_observation_information_nats']for s in scores])),
            random_probabilities=[1/3]*3,approximation='Best linear predictor covariance utility; not exact Bayes risk or mutual information')
    return result


def paired_cluster_summary(values,*,seed=BOOTSTRAP_SEED):
    """No dropping missing objects; 16 paired object differences are required."""
    x=np.asarray(values,float)
    if x.shape!=(OBJECTS,)or not np.isfinite(x).all():raise ValueError('All16 independent object clusters required')
    indices=np.random.default_rng(seed).integers(OBJECTS,size=(10000,OBJECTS))
    means=x[indices].mean(1)
    return dict(objects=OBJECTS,mean=float(x.mean()),CI95=np.quantile(means,[.025,.975]).tolist(),bootstrap_seed=seed,bootstrap_repetitions=10000)


def weighted_hypothesis_predictor(unit,inverse_amplitude,log_ratio,ratio_predict_logz,fallback):
    """Stream exact irregular hypotheses without repeating spectral sums.

    The statistic routine passes contiguous views of its unchanged unit grid.
    Views are mapped by exact memory offsets and equality, never a rounded
    parameter search. Arbitrary external coordinates use the exact fallback.
    """
    u=np.asarray(unit,float);a=np.asarray(inverse_amplitude,float);z=np.asarray(log_ratio,float)
    if u.ndim!=2 or u.shape[1]!=2 or not u.flags.c_contiguous or a.shape!=(len(u),)or z.shape!=(len(u),)or not np.isfinite(u).all()or not np.isfinite(a).all()or not np.isfinite(z).all()or np.any(a<=0):raise ValueError('Finite contiguous unit grid and positive physical amplitude nodes')
    unique,inverse=np.unique(z,return_inverse=True);tables={};address=u.__array_interface__['data'][0];rowbytes=u.strides[0]
    def predict(query,labels):
        q=np.asarray(query,float);labels=tuple(labels);start=None
        if q.ndim==2 and q.shape[1]==2 and q.flags.c_contiguous and np.shares_memory(q,u):
            offset=q.__array_interface__['data'][0]-address
            if offset>=0 and offset%rowbytes==0:
                candidate=offset//rowbytes
                if candidate+len(q)<=len(u)and np.array_equal(q,u[candidate:candidate+len(q)]):start=candidate
        if start is None:return fallback(q,labels)
        if labels not in tables:
            values=np.asarray(ratio_predict_logz(unique,labels),float)
            if values.ndim!=3 or values.shape[:2]!=(len(unique),len(labels))or not np.isfinite(values).all():raise ValueError('Finite all-mode ratio predictions')
            tables[labels]=values
        return a[start:start+len(q),None,None]*tables[labels][inverse[start:start+len(q)]]
    return predict
