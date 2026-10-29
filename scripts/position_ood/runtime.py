"""Explicit isolated campaign namespaces; legacy sharp-patch runs stay intact."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=os.environ.get('POSITION_CAMPAIGN','position-ood-v09')
BOUNDARY=os.environ.get('POSITION_BOUNDARY','sharp')
if CAMPAIGN not in ['position-ood-v09','position-ood-v10']:raise ValueError('Unknown campaign namespace')
if BOUNDARY not in ['sharp','smooth']:raise ValueError('Unknown attachment model')
if CAMPAIGN=='position-ood-v10' and BOUNDARY!='smooth':raise ValueError('v10 requires explicitly selected smooth attachment physics')
BASE=ROOT/'runs'/CAMPAIGN;OUT=ROOT/'reports'/CAMPAIGN
CONFIG=ROOT/'configs'/('position_ood_v10.json' if CAMPAIGN.endswith('v10') else 'position_ood_v09.json')

def dense_model(*args,**kwargs):
    if len(args)<2 and 'device' not in kwargs:kwargs['device']=os.environ.get('POSITION_INVERSE_DEVICE','cuda:0')
    if BOUNDARY=='smooth':
        from scripts.position_ood.smooth_fixture import SmoothDensePositionFEM
        return SmoothDensePositionFEM(*args,**kwargs)
    from scripts.position_ood.differentiable import DensePositionFEM
    return DensePositionFEM(*args,**kwargs)
