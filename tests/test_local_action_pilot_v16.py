"""Cheap source-orchestration check: selection committed before candidate access."""
import json
from types import SimpleNamespace
import numpy as np
from scripts.compositional_neural_mechanics import local_action_pilot as p

def test_worker_commits_choice_before_candidate_values(tmp_path,monkeypatch):
 folder=tmp_path/'world0';folder.mkdir();planpath=tmp_path/'plan.json';planpath.write_text('{}')
 poses=np.zeros((4,2,6));points=np.array([[0.,0.,0.]])
 (folder/'public.json').write_text(json.dumps(dict(points=points.tolist(),observed_poses=poses.tolist(),held_poses=poses[:2].tolist(),geometry={},categories={})))
 np.savez(folder/'initial.npz',displacement=points)
 for a in p.ACTIONS:np.savez(folder/(a+'.npz'),displacement=points)
 monkeypatch.setattr(p,'OUT',tmp_path);monkeypatch.setattr(p,'PLAN',planpath)
 monkeypatch.setattr(p,'guard',lambda:dict(seed=926847))
 monkeypatch.setattr(p,'build',lambda *a:SimpleNamespace())
 monkeypatch.setattr(p,'fit',lambda *a:dict(parameters=np.zeros(15),optimizer_success=True,incomplete=False))
 monkeypatch.setattr(p,'GaussNewtonMAP',lambda e:e)
 monkeypatch.setattr(p,'local_covariance',lambda *a,**k:dict(covariance=np.eye(15)))
 monkeypatch.setattr(p,'public_position_jacobian',lambda *a:dict(position_jacobian_m=np.ones((6,15)),whitened_jacobian=np.ones((3,15))))
 monkeypatch.setattr(p,'prediction',lambda *a:dict(mean=np.zeros(6),sd=np.ones(6)))
 original=np.load;reads=[]
 def guarded(path,*args,**kwargs):
  if path.name in [a+'.npz'for a in p.ACTIONS]:
   assert (folder/'selection.json').exists();reads.append(path.name)
  return original(path,*args,**kwargs)
 monkeypatch.setattr(p.np,'load',guarded);p.worker(0)
 assert len(reads)==3
 assert (folder/'complete.json').exists()
 assert len(json.loads((folder/'selection.json').read_text())['policies'])==5
