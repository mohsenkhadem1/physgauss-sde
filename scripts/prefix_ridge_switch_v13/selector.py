"""Fixed-feature guarded nodal ridge versus archived chosen SHORT configuration."""
import numpy as np
from scipy.spatial import cKDTree
from scripts.external_benchmark.evaluate import features
from scripts.contact_benchmark_v12.model_selection import _payload,_validation_window,_predict,_score,_topology_failures,rigid_nodal,CONFIGS
from scripts.contact_benchmark_v12.feasibility import inherited_elements,repair
from scripts.contact_benchmark_v12.contact_rollout import rollout

def ridge(data,locations=None):
    _payload(data);end=int(data['train_end']);F=features(data);mean=F[:end].mean(0);std=F[:end].std(0);std[std<1e-5]=1
    X=np.c_[np.ones(end),(F[:end]-mean)/std];XX=np.c_[np.ones(len(F)),(F-mean)/std]
    observed=data['observed']-data['observed_initial'][None];visible=data['observed_mask'];pred=np.zeros((len(F),observed.shape[1],3));coeffs=[]
    for i in range(observed.shape[1]):
        use=visible[:,i];design=X[use];target=observed[use,i]
        coefficient=np.linalg.solve(design.T@design+np.eye(X.shape[1])*.1,design.T@target);coeffs.append(coefficient);pred[:,i]=XX@coefficient
    locations=data['nodes'] if locations is None else np.asarray(locations)
    d,ids=cKDTree(data['observed_initial']).query(locations,k=min(6,len(data['observed_initial'])))
    if d.ndim==1:d=d[:,None];ids=ids[:,None]
    weights=np.exp(-((d/np.maximum(d[:,-1,None],1e-9))**2)*3)/np.maximum(d,1e-6);weights/=weights.sum(1,keepdims=True)
    result=locations[None]+np.einsum('pk,tpkc->tpc',weights,pred[:,ids])
    return result,dict(mean=mean,std=std,coefficients=np.asarray(coeffs))

def guarded(data,anchor,scene,elements):
    proposal,_=ridge(data);prediction,meta=repair(data,anchor,proposal,elements=elements)
    if data['dataset']=='SoftVTBench':
        if scene is None:raise ValueError('Same static scene required')
        prediction,contact=rollout(data,prediction,scene);meta={'repair':meta,'contact':contact}
    bad=_topology_failures(data,prediction,0,len(prediction),elements)
    if bad:prediction=prediction.copy();prediction[bad]=np.nan
    return prediction,{'guard':meta,'post_contact_invalid_frames':bad}

def select(full,short,current_short,current_full,field_free_short,old_metadata,old_forecast,scene=None,elements=None):
    """No evaluator input. Fixed fallback is bitwise archived forecast including NaNs."""
    default=np.asarray(old_forecast).copy();meta={'scope':'Exploratory guarded nodal empirical ridge, not coupled physical transfer','switched':False,'reason':'not evaluated','old_validation_rmse_m':None,'ridge_validation_rmse_m':None}
    try:
        _payload(full);_payload(short);end=int(full['train_end']);cut=int(short['train_end'])
        if not 2<=cut<end:raise ValueError('Invalid independent validation cut')
        np.testing.assert_array_equal(full['observed'][:cut],short['observed']);np.testing.assert_array_equal(full['observed_mask'][:cut],short['observed_mask'])
        for name in ('nodes','observed_initial','observation_matrix'):np.testing.assert_array_equal(full[name],short[name])
        expected=(len(full['times']),)+np.shape(full['nodes'])
        for value in (current_short,current_full,field_free_short,old_forecast):
            if np.shape(value)!=expected:raise ValueError('Cached forecast shape mismatch')
        data=_validation_window(short,end);elements=inherited_elements(data) if elements is None else elements
        chosen=old_metadata['selected_configuration'];config=(chosen['family'],chosen['method'],chosen['strength'])
        if config[0] not in ('current','field_free','rigid') or (config[1],config[2]) not in CONFIGS or (config[0]=='rigid' and config[1:]!=('none',0.)):raise ValueError('Invalid archived selected configuration')
        anchors={'current':np.asarray(current_short)[:end],'field_free':np.asarray(field_free_short)[:end]}
        anchors['rigid']=rigid_nodal(data) if config[0]=='rigid' else None
        old_validation,_=_predict(data,anchors[config[0]],config,scene,elements)
        old_error,old_bad=_score(full,old_validation,data,cut,end)
        old_topology=_topology_failures(data,old_validation,cut,end,elements)
        if old_topology:old_error=float('inf')
        if 'validation_finite' in chosen:
            expected_error=chosen['validation_rmse_m'] if chosen['validation_finite'] else float('inf')
            if not np.isclose(old_error,expected_error,rtol=1e-10,atol=1e-12):raise ValueError('Archived SHORT validation reconstruction mismatch')
        new_validation,_=guarded(data,anchors['current'],scene,elements)
        new_error,new_bad=_score(full,new_validation,data,cut,end)
        meta.update(validation_cut=cut,full_prefix_end=end,old_configuration=chosen,old_validation_rmse_m=old_error if np.isfinite(old_error) else None,ridge_validation_rmse_m=new_error if np.isfinite(new_error) else None,old_validation_failures=sorted(set(old_bad+old_topology)),ridge_validation_failures=new_bad)
        if not (np.isfinite(new_error) and new_error<.98*old_error):meta['reason']='No strict2% prefix improvement';return default,meta
    except Exception as error:
        meta['reason']='Unavailable validation; exact archived fallback';meta['validation_error']=repr(error);return default,meta
    meta.update(switched=True,reason='Strict2% observed-prefix improvement')
    try:
        prediction,final=guarded(full,np.asarray(current_full),scene,elements);meta['final']=final
    except Exception as error:
        prediction=np.full_like(default,np.nan,dtype=float);meta['final_error']=repr(error)
    meta['failed_full_frames']=np.flatnonzero(~np.isfinite(prediction).all(axis=(1,2))).tolist()
    return prediction,meta
