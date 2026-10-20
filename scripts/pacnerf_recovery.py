"""Restore frozen observation constants without weakening checkpoint identity.

CPU implementations can differ in the last few bits of an image correlation
reduction. Resume uses the ORIGINAL saved correlations, not their recomputation.
All other model, input, source and kernel fields must still match exactly.
"""
import copy
import math


CORRELATION_ROUNDOFF_BOUND=64*2.220446049250313e-16


def differences(old,new,path=''):
    if isinstance(old,dict) and isinstance(new,dict):
        result=[]
        for key in sorted(old.keys()|new.keys()):
            name=f'{path}.{key}' if path else key
            if key not in old or key not in new:result.append(name)
            else:result.extend(differences(old[key],new[key],name))
        return result
    if isinstance(old,list) and isinstance(new,list):
        if len(old)!=len(new):return [path]
        return [p for i,(a,b) in enumerate(zip(old,new)) for p in differences(a,b,f'{path}[{i}]')]
    return [] if type(old)==type(new) and old==new else [path]


def restore_correlations(model,saved_model):
    computed=model.specification()
    frozen=copy.deepcopy(saved_model);candidate=copy.deepcopy(computed)
    old=frozen.pop('rgb_spatial_correlation',None)
    new=candidate.pop('rgb_spatial_correlation',None)
    other=differences(frozen,candidate,'model')
    if other:raise ValueError('Recovery model changed: '+', '.join(other[:20]))
    if not isinstance(old,list) or not isinstance(new,list) or len(old)!=2 or len(new)!=2:
        raise ValueError('Recovery correlations must contain two frozen values')
    for a,b in zip(old,new):
        if (type(a) not in (int,float) or type(b) not in (int,float)
            or not math.isfinite(a) or not math.isfinite(b) or not 0<=a<=.95 or not 0<=b<=.95
            or abs(a-b)>CORRELATION_ROUNDOFF_BOUND):
            raise ValueError('Recomputed observation correlation differs beyond float64 roundoff')
    model.rho_x,model.rho_y=map(float,old)
    if differences(saved_model,model.specification()):raise ValueError('Frozen observation constants were not restored exactly')
    return {'recomputed_correlations':new,'frozen_correlations':old,
        'maximum_recomputation_difference':max(abs(a-b) for a,b in zip(old,new)),
        'roundoff_bound':CORRELATION_ROUNDOFF_BOUND,
        'policy':'Restore original saved constants exactly; no tolerance applied to source/input/kernel/model identity'}


def restore_settings(model,saved,current):
    old=copy.deepcopy(saved);new=copy.deepcopy(current)
    old.pop('model',None);new.pop('model',None)
    changed=differences(old,new)
    if changed:raise ValueError('Target or protocol changed: '+', '.join(changed[:20]))
    proof=restore_correlations(model,saved['model'])
    restored=copy.deepcopy(current);restored['model']=model.specification()
    if differences(saved,restored):raise ValueError('Recovery settings still differ after restoring frozen constants')
    return restored,proof
