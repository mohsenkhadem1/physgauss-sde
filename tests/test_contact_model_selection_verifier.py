import copy
import numpy as np
import pytest
from scripts.contact_benchmark_v12.model_selection_verify import verify_decision
from scripts.contact_benchmark_v12.model_selection import CONFIGS


def fixture(values):
    configs=[(f,m,s) for f in ['current','field_free'] for m,s in CONFIGS]+[('rigid','none',0.)]
    rows=[dict(family=f,method=m,strength=s,validation_finite=np.isfinite(v).item(),validation_rmse_m=v if np.isfinite(v) else None,failed_validation_frames=[] if np.isfinite(v) else [3]) for (f,m,s),v in zip(configs,values)]
    best=int(np.argmin(values));switch=bool(np.isfinite(values[best]) and values[best]<.98*values[0]);chosen=best if switch else 0
    winners={f:min([i for i,c in enumerate(configs) if c[0]==f],key=lambda i:values[i]) for f in ['current','field_free','rigid']}
    pred={f'model_selection_{f}':np.full((2,4,3),i,dtype=float) for i,f in enumerate(winners)}
    default=np.full((2,4,3),9.)
    pred['model_selection_selected']=pred['model_selection_'+configs[chosen][0]].copy() if switch else default.copy()
    meta=dict(default_configuration=dict(family='current',method='none',strength=0.),validation=rows,switched_from_previous=switch,selected_configuration=rows[chosen],family_winners={f:rows[i] for f,i in winners.items()},final_predictions={'selected':{'exact_previous_guarded_prediction':not switch}})
    return meta,pred,default


def test_decision_rebuild_and_exact_default_preservation():
    meta,pred,default=fixture([1.]*15)
    assert not verify_decision(meta,pred,default)['switched']
    pred['model_selection_selected'][0,0,0]+=1e-12
    with pytest.raises(AssertionError):verify_decision(meta,pred,default)


def test_decision_rejects_wrong_threshold_and_wrong_winner_output():
    values=[1.]*15;values[7]=.97
    meta,pred,default=fixture(values)
    assert verify_decision(meta,pred,default)['selected_family']=='field_free'
    broken=copy.deepcopy(meta);broken['switched_from_previous']=False
    with pytest.raises(AssertionError):verify_decision(broken,pred,default)
    pred['model_selection_selected'][0,0,0]+=1
    with pytest.raises(AssertionError):verify_decision(meta,pred,default)
