"""Read-only development catalogue checks; no fresh data or simulations."""
import copy
import numpy as np
import pytest
from scripts.compositional_dynamics_v17 import cohort_shared_fit as m

def recipe():
 return dict(catalogue_plan='reports/compositional-dynamics-v17/matched-material-fit-plan.json',catalogue_results='reports/compositional-dynamics-v17/matched-material-fit-results.json')

def test_catalogue_exact_identity_and_per_world_loss():
 metadata,obs,visibility=m.load_public(m.base.PUBLIC)
 _,rows=m.catalogue(recipe(),metadata)
 assert len(rows)==54
 row=next(r for r in rows if r['completed'])
 with np.load(m.ROOT/row['path']/'prediction.npz')as a:prediction=a['positions_m']
 old=m.base.observed_loss(prediction,obs,visibility)[0]
 changed=obs.copy();changed[1:]+=.001
 new=m.base.observed_loss(prediction,changed,visibility)[0]
 assert old!=new
 # Source candidate loss is deliberately not used for the fresh observation.
 assert np.isfinite(new)

@pytest.mark.parametrize('key,value',[('total_mass_kg',.28),('gravity_vector_m_s2',[0,0,-9.81]),('frame_dt_s',.02),('initial_velocity_m_s',[0,0,-.1])])
def test_cache_rejects_changed_forward_contract(key,value):
 metadata,_,_=m.load_public(m.base.PUBLIC);changed=copy.deepcopy(metadata);changed[key]=value
 with pytest.raises(ValueError,match='Catalogue public physics'):m.catalogue(recipe(),changed)

def test_bad_protocol_rejected_before_public_access(tmp_path):
 p=tmp_path/'p';f=tmp_path/'f';a=tmp_path/'a'
 p.write_text('{}');f.write_text('{}');a.write_text('{}')
 with pytest.raises(ValueError,match='digest'):m.guard(dict(protocol_path=str(p),freeze_path=str(f),approval_path=str(a),protocol_sha='wrong',freeze_sha='wrong'))

def test_explicit_failed_export_and_missing_status(tmp_path):
 bindings=dict(protocol_sha='p',freeze_sha='f')
 m.write(tmp_path/'complete.json',dict(completed=False,world_id='world-0',protocol_sha256='p',method_freeze_sha256='f'))
 assert m.public_status(tmp_path,'world-0',bindings)['completed'] is False
 with pytest.raises(ValueError,match='identity'):m.public_status(tmp_path,'world-1',bindings)
 with pytest.raises(FileNotFoundError):m.public_status(tmp_path/'missing','world-0',bindings)

def test_failed_forecast_is_terminal_without_prediction(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path)
 monkeypatch.setattr(m,'require_runtime',lambda:None)
 monkeypatch.setattr(m,'guard',lambda bindings:({'generator_settings':{'world_ids':['world-0']}},{'forecast_root':'predictions'}))
 def fail(*args):raise FileNotFoundError('No committed available fit')
 monkeypatch.setattr(m,'_forecast',fail)
 m.forecast({'protocol_sha':'p'},'world-0','forecast-80mm')
 import json
 path=tmp_path/'predictions/world-0/forecast-80mm'
 record=json.loads((path/'prediction-commit.json').read_text())
 assert record['terminal'] and record['completed'] is False and not (path/'trajectory.npz').exists()
 with pytest.raises(ValueError,match='Preserve'):m.forecast({'protocol_sha':'p'},'world-0','forecast-80mm')
