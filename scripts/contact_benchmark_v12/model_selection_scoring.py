"""Evaluator-only handling of unavailable initial manual correspondences."""
import numpy as np
from scripts.external_benchmark.evaluate import score


def safe_score(data,pred,start,end,selected=None):
    if data['dataset']!='PhysTwin' or np.isfinite(pred[0]).all():
        return score(data,pred,start,end,selected)
    ordinary=dict(data,dataset='PhysTwin_manual_correspondence_unavailable')
    result=score(ordinary,pred,start,end,selected)
    result['summary']['manual_track_mean_mm']=None
    result['manual_correspondence_unavailable']=dict(reason='Initial prediction contains nonfinite coordinates; initial nearest-neighbor manual correspondence is undefined',initial_prediction_finite=False,prediction_modified=False)
    return result
