"""Production interface for the development-selected original-anchor slip law."""
import numpy as np
from scripts.benchmark_improvement_v14.softvt_slip import predict as simple_slip

REQUIRED = frozenset(('dataset','nodes','train_end','observed','observed_mask','observed_initial','observation_matrix','times','grasp','attachment_active','gripper_command','controller_targets'))
ALLOWED = REQUIRED | frozenset(('name','support_geometry','attachment_weights','floor','controls','source','gripper_width'))


def predict(data, base_prediction=None, config=None):
    """Fixed law only. Baseline is ORIGINAL v12 contact_scene_state nodal output."""
    config={} if config is None else config
    if set(data)-ALLOWED or REQUIRED-set(data):
        raise ValueError('Unexpected/missing predictor inputs; evaluation targets forbidden')
    if set(config)-{'scene','window'} or config.get('window',8)!=8 or 'scene' not in config:
        raise ValueError('Frozen config requires scene and window8 only')
    end=int(data['train_end']);base=np.asarray(base_prediction,float)
    if data['dataset']!='SoftVTBench' or base.shape!=(len(data['times']),len(data['nodes']),3):
        raise ValueError('Wrong dataset or nodal baseline shape')
    if not 2<=end<=len(base) or len(data['observed'])!=end or len(data['observed_mask'])!=end:
        raise ValueError('Strict observed prefix required')
    if not np.isfinite(base).all():
        return base.copy(),dict(method='simple_slip_original_anchor',fallback='Original baseline contains nonfinite frames; preserved exactly',fallback_to_original=True)
    try:
        result,meta=simple_slip(data,base,dict(scene=config['scene'],window=8))
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        return base.copy(),dict(method='simple_slip_original_anchor',fallback=str(error),fallback_to_original=True)
    if result.shape!=base.shape or not np.isfinite(result).all():
        return base.copy(),dict(method='simple_slip_original_anchor',fallback='Nonfinite or malformed proposed forecast',fallback_to_original=True)
    meta=dict(meta,anchor='original frozen v12 contact_scene_state',fallback_to_original=bool(meta.get('fallback')),development_selected=True)
    return result,meta
