"""Independent NumPy reconstruction of the discarded-warmup proposal reference."""
import numpy as np


def conditional_transform(precision, core):
    d=len(precision);s=np.arange(2,core);p=np.array([0,1,*range(core,d)])
    ss=precision[np.ix_(s,s)];cross=precision[np.ix_(s,p)]
    regression=-np.linalg.solve(ss,cross)
    schur=precision[np.ix_(p,p)]+cross.T@regression
    transform=np.zeros_like(precision)
    parameter=np.linalg.solve(np.linalg.cholesky(schur).T,np.eye(len(p)))
    transform[np.ix_(p,p)]=parameter
    transform[np.ix_(s,p)]=regression@parameter
    transform[np.ix_(s,s)]=np.linalg.solve(np.linalg.cholesky(ss).T,np.eye(len(s)))
    return transform


def verify_reference(reference,anchor,learning,core,settings):
    config=settings['reference_learning'];protocol=learning['protocol']
    expected_seed=240001+settings['case_index']+config['seed_offset']
    if (protocol['seed']!=expected_seed or protocol['warmup']!=config['warmup']
        or learning['completed_warmup']!=config['warmup']
        or learning['retained']!=config['draws_per_chain']
        or protocol['sampler']!='conditional-state-hmc-parameter-ess-v1'):
        raise ValueError('Reference-learning sampling protocol differs')
    white=learning['white'].numpy().reshape(-1,len(anchor['mean']))
    transform=conditional_transform(anchor['precision'].numpy(),core)
    mean=anchor['mean'].numpy()+transform@white.mean(0)
    shrinkage=config['shrinkage']
    covariance=transform@((1-shrinkage)*np.cov(white.T)+shrinkage*np.eye(white.shape[1]))@transform.T
    precision=np.linalg.inv(covariance)
    last_native=anchor['mean'].numpy()+learning['position'].numpy()@transform.T
    initial=np.linalg.solve(conditional_transform(precision,core),(last_native-mean).T).T
    errors={}
    for key,expected in [('sampling_mean',mean),('sampling_precision',precision),('initial_white',initial)]:
        actual=reference[key].numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-8,err_msg=key)
        errors[key]=float(np.max(np.abs(actual-expected)))
    for key in ['mean','precision','anchor']:
        np.testing.assert_array_equal(reference[key].numpy(),anchor[key].numpy())
    if not reference['reference_learning']['future_geometry_used'] is False:
        raise ValueError('Reference learning must not use future geometry')
    return {'all_pass':True,'discarded_reference_draws':len(white),'maximum_absolute_errors':errors,
        'scope':'NumPy replay of sampling mean, covariance and initial-state mapping; fitted mode unchanged; reference-learning draws excluded from posterior summaries'}
