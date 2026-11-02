import copy
import numpy as np
import pytest
from scripts.compositional_neural_mechanics import train_direct_campaign as c

def test_exact_scene_action_and_query_membership():
 dataset=dict(material_query=c.grid(7).tolist(),scenes=[dict(role='train'if i<54 else'validation',scene={'name':f'scene-{i}'},actions=[{}for _ in range(6)])for i in range(58)])
 roles=c.membership(dataset);assert len(roles['train'])==54 and len(roles['validation'])==4
 bad=copy.deepcopy(dataset);bad['scenes'][0]['role']='reserved'
 with pytest.raises(ValueError):c.membership(bad)
 bad=copy.deepcopy(dataset);bad['scenes'][0]['actions'].pop()
 with pytest.raises(ValueError):c.membership(bad)
 bad=copy.deepcopy(dataset);bad['material_query']=c.grid(5).tolist()
 with pytest.raises(ValueError):c.membership(bad)

def test_no_training_with_missing_declared_records(monkeypatch,tmp_path):
 plan=dict(split_manifest={'train':[],'validation':[]})
 monkeypatch.setattr(c,'guard',lambda p:(plan,{}));monkeypatch.setattr(c,'OUT',tmp_path)
 def missing(*args,**kwargs):raise ValueError('All348records required')
 monkeypatch.setattr(c,'records',missing)
 def forbidden(*args,**kwargs):raise AssertionError('Training must never start')
 monkeypatch.setattr(c,'train_direct',forbidden)
 with pytest.raises(ValueError,match='All348'):c.run(tmp_path/'plan.json')
