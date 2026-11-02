import json
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.compositional_neural_mechanics import unknown_action_campaign as p
from scripts.compositional_neural_mechanics import unknown_action_score as s

def test_sixth_world_choice_precedes_observed_candidate(tmp_path,monkeypatch):
 folder=tmp_path/'world5';folder.mkdir();planpath=tmp_path/'plan.json';planpath.write_text('{}');(tmp_path/'generation-complete.json').write_text('{}')
 poses=np.zeros((4,2,6));points=np.zeros((26,3))
 (folder/'public.json').write_text(json.dumps(dict(points=points.tolist(),observed_poses=poses.tolist(),held_poses=poses[:2].tolist(),geometry={},categories={})))
 for a in ('initial',*p.ACTIONS):np.savez(folder/(a+'.npz'),displacement=points)
 monkeypatch.setattr(p,'B',tmp_path);monkeypatch.setattr(p,'PLAN',planpath);monkeypatch.setattr(p,'input_guard',lambda world:dict(random_policy_seed=926967))
 monkeypatch.setattr(p,'build',lambda *a:SimpleNamespace());monkeypatch.setattr(p,'fit',lambda *a:dict(parameters=np.zeros(15),optimizer_success=True,incomplete=False))
 monkeypatch.setattr(p,'GaussNewtonMAP',lambda e:e);monkeypatch.setattr(p,'local_covariance',lambda *a,**k:dict(covariance=np.eye(15)))
 monkeypatch.setattr(p,'public_position_jacobian',lambda *a:dict(position_jacobian_m=np.ones((156,15)),whitened_jacobian=np.ones((78,15))))
 monkeypatch.setattr(p,'prediction',lambda *a:dict(mean=np.zeros(156),sd=np.ones(156)))
 load=np.load;seen=[]
 def guarded(path,*a,**k):
  if path.name in [x+'.npz'for x in p.ACTIONS]:assert(folder/'shared_laws_fitted/selection.json').exists();seen.append(path.name)
  return load(path,*a,**k)
 monkeypatch.setattr(p.np,'load',guarded);p.worker(5,'shared_laws_fitted');assert len(seen)==3
 assert(folder/'shared_laws_fitted/complete.json').exists()
 first=json.loads((folder/'shared_laws_fitted/selection.json').read_text());original_prediction=(folder/'shared_laws_fitted/forecasts.npz').read_bytes()
 (folder/'shared_laws_fitted').rename(folder/'first-preserved')
 private=folder/'private';private.mkdir();np.savez(private/'held-truth.npz',parameters=np.full(15,999.),held0=np.full((26,3),-999.));(private/'scene.json').write_text('{"hidden_parameter":999}')
 p.worker(5,'shared_laws_fitted')
 assert json.loads((folder/'shared_laws_fitted/selection.json').read_text())==first
 with load(folder/'first-preserved/forecasts.npz')as old,load(folder/'shared_laws_fitted/forecasts.npz')as new:
  assert set(old.files)==set(new.files)
  assert all(np.array_equal(old[k],new[k])for k in old.files)

def test_scorer_refuses_early_before_any_truth(tmp_path,monkeypatch):
 monkeypatch.setattr(s,'B',tmp_path);monkeypatch.setattr(s,'guard',lambda:{})
 def forbidden(*a,**k):raise AssertionError('Truth must not be opened')
 monkeypatch.setattr(s.np,'load',forbidden)
 with pytest.raises(FileNotFoundError):s.score()
