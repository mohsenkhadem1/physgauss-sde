"""Independent input-isolation checks; no future benchmark errors are computed."""
import copy
import pickle
from unittest.mock import patch
import numpy as np
import h5py
import pytest
from scripts.contact_benchmark_v12.data import load, trim_prefix, predictor_payload
from scripts.contact_benchmark_v12.experiment import predict
from scripts.contact_benchmark_v12.scene import load_scene, StaticScene


def same(a,b):
    if isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


class PoisonNode:
    def __init__(self,node,cut):self.node=node;self.cut=cut
    def __getattr__(self,key):return getattr(self.node,key)
    def __iter__(self):return iter(self.node)
    def __enter__(self):return self
    def __exit__(self,*args):self.node.close()
    def __getitem__(self,key):
        if isinstance(self.node,h5py.Group):return PoisonNode(self.node[key],self.cut)
        value=self.node[key]
        name=self.node.name
        if '/states/' in name or '/obs/' in name:
            forbidden=('/rigid_object/' in name or '/nodal_pos_w' in name or
                       any(s in name for s in ['force','torque','tactile']))
            if forbidden:
                value=np.asarray(value).copy()
                if value.ndim and value.shape[0]>self.cut:value[self.cut:]=1e6
        return value


def simple_mesh(*args,**kwargs):
    return np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]]),np.array([[0,1,2],[0,2,3]]),{'fixture':'fixed public-geometry surrogate'}


def fake_base(data):
    # Input-only shape proposal, deliberately independent of hidden truth.
    return np.mean(data['controller_targets'],axis=1)


def predict_payload(data,validation,scene):
    return predict(predictor_payload(data),fake_base(data),fake_base(validation),predictor_payload(validation),scene)


@pytest.mark.parametrize('subset',['object-soft','spatial-soft'])
def test_real_softvt_future_targets_rigid_states_and_validation_isolated(subset):
    spec=dict(subset=subset,task=0,episode=1)
    data=load(spec);cut=max(3,int(data['train_end']*.7));short=trim_prefix(data,cut)
    real_file=h5py.File
    # Full forecast: future target changes do not alter payload or candidate outputs.
    with patch('h5py.File',lambda *a,**kw:PoisonNode(real_file(*a,**kw),data['train_end'])):
        poisoned=load(spec)
        with patch('scripts.contact_benchmark_v12.scene.asset_mesh',simple_mesh):scene_b=load_scene(predictor_payload(poisoned),spacing=.01)
    with patch('scripts.contact_benchmark_v12.scene.asset_mesh',simple_mesh):scene_a=load_scene(predictor_payload(data),spacing=.01)
    assert not np.array_equal(data['evaluation']['points'],poisoned['evaluation']['points'])
    same(predictor_payload(data),predictor_payload(poisoned));same(scene_a.metadata(),scene_b.metadata())
    same(predict_payload(data,short,scene_a),predict_payload(poisoned,trim_prefix(poisoned,cut),scene_b))
    # CV adaptation must recalibrate from the shortened prefix even if original
    # adapter temporarily saw later prefix observations while loading the file.
    with patch('h5py.File',lambda *a,**kw:PoisonNode(real_file(*a,**kw),cut)):
        poisoned_short=load(spec,validation=True)
    same(predictor_payload(short),predictor_payload(poisoned_short))


def test_real_phystwin_future_tracks_manual_markers_and_cv_isolated():
    data=load(dict(case='single_lift_sloth'));cut=int(data['train_end']*.7)
    short=trim_prefix(data,cut);real_load=pickle.load
    def poisoned_load(f,*args,**kwargs):
        value=real_load(f,*args,**kwargs)
        if str(f.name).endswith('final_data.pkl'):value['object_points'][cut:]=1e6
        elif str(f.name).endswith('gt_track_3d.pkl'):value[:]=1e6
        return value
    with patch('pickle.load',poisoned_load):poisoned_short=load(dict(case='single_lift_sloth'),validation=True)
    same(predictor_payload(short),predictor_payload(poisoned_short))
    a,b=predict_payload(data,short,None)
    tampered=copy.deepcopy(data);tampered['evaluation']={'points':np.full((3,),np.nan),'manual':1e99}
    aa,bb=predict_payload(tampered,poisoned_short,None)
    same(a,aa);same(b,bb)


def test_candidate_entry_rejects_evaluator_payload():
    with pytest.raises(ValueError,match='Future truth'):
        predict({'evaluation':{}},None,None,{},StaticScene())
