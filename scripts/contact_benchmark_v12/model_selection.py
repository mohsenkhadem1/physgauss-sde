"""Exploratory prefix-only family selection; frozen v12 models remain unchanged."""
import numpy as np
from scripts.contact_benchmark_v12.object_frame import adapt,control_frames,_observed_prediction
from scripts.contact_benchmark_v12.feasibility import repair,inherited_elements,jacobians
from scripts.contact_benchmark_v12.contact_rollout import rollout
from scripts.external_benchmark.data import rigid_map

CONFIGS=[('none',0.)]+[(method,strength) for method in ('pose','residual') for strength in (.25,.5,1.)]


def _payload(data):
    if 'evaluation' in data:raise ValueError('Evaluator target dictionary forbidden')
    end=int(data['train_end'])
    if len(data['observed'])!=end or len(data['observed_mask'])!=end:raise ValueError('Payload observations must end exactly at declared cut')


def _validation_window(data,end):
    """Ignore all postvalidation predictions/inputs, including future failures."""
    out=dict(data)
    for key in ['times','controller_targets','attachment_active','controls','gripper_width','gripper_command']:
        if key in out:out[key]=out[key][:end].copy()
    return out


def rigid_nodal(data):
    """Calibrate nodal rigid transport from observed prefix only, no evaluator."""
    _payload(data);nodes=np.asarray(data['nodes']);frames=control_frames(data);end=data['train_end'];last=end-1
    valid=np.asarray(data['observed_mask'][last],bool)&np.isfinite(data['observed'][last]).all(1)
    if valid.sum()<3:raise ValueError('Rigid calibration needs three observed points')
    R,b=frames[last];local=(data['observed'][last,valid]-b)@R.T
    offset,translation=rigid_map(data['observed_initial'][valid],local);shape=nodes@offset+translation
    result=[]
    for t,(R,b) in enumerate(frames):
        if np.asarray(data['attachment_active'][t]).sum()>0:points=shape@R+b
        elif data['dataset']=='SoftVTBench' and t>data['grasp'] and len(result)>1:
            dt=float(data['times'][t]-data['times'][t-1]);velocity=(result[-1].mean(0)-result[-2].mean(0))/dt
            points=result[-1]+velocity*dt+[0,0,-9.81*dt*dt];points[:,2]+=max(0.,data['floor']-points[:,2].min())
        else:points=nodes.copy()
        result.append(points)
    result=np.asarray(result);result[0]=nodes
    return result


def _predict(data,anchor,config,scene,elements):
    family,method,strength=config
    if family=='rigid':proposal=anchor.copy();adapter=dict(method='prefix_calibrated_nodal_rigid')
    else:proposal,adapter=adapt(data,anchor,method=method,strength=strength)
    guarded,guard=repair(data,anchor,proposal,elements=elements)
    contact=None
    if data['dataset']=='SoftVTBench':
        if scene is None:raise ValueError('Same static scene required for all SoftVT candidates')
        guarded,contact=rollout(data,guarded,scene)
    return guarded,dict(adapter=adapter,guard=guard,contact=contact)


def _topology_failures(data,prediction,start,stop,elements):
    failed=[]
    for t in range(start,stop):
        if not np.isfinite(prediction[t]).all():continue
        J=jacobians(data['nodes'],prediction[t],elements)
        if not (np.isfinite(J)&(J>.03+1e-8)).all():failed.append(t)
    return failed


def _score(full,prediction,short,cut,end):
    failed=np.flatnonzero(~np.isfinite(prediction[cut:end]).all(axis=(1,2)))+cut
    target=full['observed'][cut:end];valid=np.asarray(full['observed_mask'][cut:end],bool)&np.isfinite(target).all(2)
    if len(failed) or not valid.any():return float('inf'),failed.tolist()
    projected=_observed_prediction(short,prediction[cut:end])
    return float(np.sqrt(np.mean((projected[valid]-target[valid])**2))),[]


def select_model(data,validation_data,families,default_metadata,default_guarded_prediction,scene=None):
    """Return ({selected,current,field_free,rigid}, audit metadata).

    families={current:{full:TxNx3,validation:TxNx3},field_free:{...}}.
    default_metadata is original object_frame adapter metadata, or its enclosing
    prediction-metadata dictionary. Original prefix-selected configuration is
    reconstructed on SHORT current-family fit to establish default validation.
    No supplied full-prefix adapter predictions enter candidate validation.
    Fixed ordering and strict2% improvement retain exact prior guarded output on
    ties/all failures. A new winning full-prefix failure stays explicit NaN.
    Family winners are separately refitted for reporting, never future-selected.
    """
    _payload(data);_payload(validation_data)
    end=int(data['train_end']);cut=int(validation_data['train_end'])
    if not 2<=cut<end:raise ValueError('Independent validation cut must precede full observed end')
    np.testing.assert_array_equal(data['nodes'],validation_data['nodes'])
    np.testing.assert_array_equal(data['observed_initial'],validation_data['observed_initial'])
    np.testing.assert_array_equal(data['observed'][:cut],validation_data['observed'])
    np.testing.assert_array_equal(data['observed_mask'][:cut],validation_data['observed_mask'])
    if set(families)!= {'current','field_free'}:raise ValueError('Both prespecified independent fit families required')
    for family,values in families.items():
        if set(values)!= {'full','validation'}:raise ValueError('Family needs full and independent validation predictions')
        for variant,points in values.items():
            if np.shape(points)!=(len(data['times']),)+np.shape(data['nodes']):raise ValueError(f'Wrong trajectory shape {family}/{variant}')
    default=np.asarray(default_guarded_prediction)
    if default.shape!=np.shape(families['current']['full']):raise ValueError('Default prediction shape mismatch')
    dm=default_metadata.get('object_frame',default_metadata);method=dm.get('method','none');strength=float(dm.get('strength',0. if method=='none' else 1.))
    if (method,strength) not in CONFIGS:raise ValueError('Default configuration outside frozen adapter shortlist')
    default_config=('current',method,strength)
    configs=[default_config]+[(f,m,a) for f in ['current','field_free'] for m,a in CONFIGS if (f,m,a)!=default_config]+[('rigid','none',0.)]
    short=_validation_window(validation_data,end);elements=inherited_elements(short)
    anchors={family:np.asarray(values['validation'])[:end] for family,values in families.items()}
    try:anchors['rigid']=rigid_nodal(short)
    except Exception:anchors['rigid']=np.full_like(anchors['current'],np.nan)
    rows=[];scores=[]
    for config in configs:
        family,method,strength=config;row=dict(family=family,method=method,strength=strength)
        try:
            prediction,meta=_predict(short,anchors[family],config,scene,elements)
            value,failed=_score(data,prediction,short,cut,end)
            topology=_topology_failures(short,prediction,cut,end,elements)
            if topology:value=float('inf')
            row.update(failed_validation_frames=sorted(set(failed+topology)),failed_topology_frames=topology)
        except Exception as error:value=float('inf');row.update(error=f'{type(error).__name__}: {error}',failed_validation_frames=list(range(cut,end)))
        row['validation_rmse_m']=value if np.isfinite(value) else None;row['validation_finite']=bool(np.isfinite(value));rows.append(row);scores.append(value)
    best=int(np.argmin(scores));switch=bool(np.isfinite(scores[best]) and scores[best]<.98*scores[0]);chosen=best if switch else 0
    winners={family:min([i for i,c in enumerate(configs) if c[0]==family],key=lambda i:scores[i]) for family in ['current','field_free','rigid']}
    predictions={};final_meta={};cache={}
    def full_prediction(index):
        config=configs[index]
        if config in cache:return cache[config]
        family=config[0]
        try:
            anchor=rigid_nodal(data) if family=='rigid' else np.asarray(families[family]['full'])
            result,metadata=_predict(data,anchor,config,scene,elements)
            topology=_topology_failures(data,result,0,len(result),elements)
            if topology:
                result=result.copy();result[topology]=np.nan;metadata['post_contact_invalid_frames']=topology
        except Exception as error:result=np.full_like(default,np.nan,dtype=float);metadata=dict(error=f'{type(error).__name__}: {error}',failed_full_prediction=True)
        cache[config]=(result,metadata);return result,metadata
    for family,index in winners.items():
        if np.isfinite(scores[index]):predictions[family],final_meta[family]=full_prediction(index)
        else:predictions[family]=np.full_like(default,np.nan,dtype=float);final_meta[family]=dict(failed_family_selection=True,reason='All validation configurations failed; no family winner')
    if switch:predictions['selected'],final_meta['selected']=full_prediction(chosen)
    else:predictions['selected']=default.copy();final_meta['selected']=dict(exact_previous_guarded_prediction=True,reason='No strict2% validation improvement' if np.isfinite(scores[0]) else 'All candidate validation scores failed')
    audit=dict(scope='Exploratory model-class selection; not frozen v12 primary',validation_cut=cut,full_prefix_end=end,
      validation_source='Independent SHORT-fit payload and per-family SHORT-fit base only; withheld permitted prefix observations score candidates',
      default_configuration=dict(family=default_config[0],method=default_config[1],strength=default_config[2]),
      selected_configuration=rows[chosen],switched_from_previous= switch,validation=rows,
      family_winners={family:rows[index] for family,index in winners.items()},final_predictions=final_meta,
      selection_rule='Fixed default-first order; strict candidate RMSE <0.98*default RMSE; nonfinite validation frame => infinite; allfailed => exactoldguarded',
      caveats=['Local-volume guard is not elastic equilibrium','Static scene contact freezes shape after separation','Known future positions, no force/material labels','Standalone family results are reporting ablations, not future-selected'])
    return predictions,audit
