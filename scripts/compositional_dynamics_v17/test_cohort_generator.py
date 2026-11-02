import json
import pytest
from scripts.compositional_dynamics_v17 import cohort_generator as g

def test_prepare_never_draws_or_generates(tmp_path,monkeypatch):
 monkeypatch.setattr(g,'PROPOSAL',tmp_path/'proposal.json');monkeypatch.setattr(g,'RAW',tmp_path/'never-created')
 def forbidden(*a,**kw):raise AssertionError('No private draws allowed during preparation')
 monkeypatch.setattr(g.np.random,'default_rng',forbidden);g.prepare();p=json.loads(g.PROPOSAL.read_text())
 assert p['method_recipe']is None and not g.RAW.exists()
 assert p['generator_settings']['expected_jobs']==20

def test_declared_twenty_job_inventory_without_draws():
 s=dict(world_ids=['a','b','c','d'],roles=[dict(name='calibration',clearance_m=.01,duration_s=.2),dict(name='forecast-40mm',clearance_m=.04,duration_s=.3),dict(name='forecast-80mm',clearance_m=.08,duration_s=.3)],dt_s=25e-6,half_dt_s=12.5e-6,expected_jobs=20)
 worlds=[dict(name=n,family=f)for n,f in zip(s['world_ids'],['yeoh','mooney_rivlin','yeoh','mooney_rivlin'])]
 jobs,subset=g.jobs(s,worlds)
 assert len(jobs)==20 and subset==['b','a']
 assert sum(j['kind']=='primary'for j in jobs)==12
 assert sum(j['n']==6 for j in jobs)==6
 assert all(j['world_id']in subset for j in jobs if j['kind']!='primary')

def test_guard_rejects_before_any_generation(tmp_path):
 p=tmp_path/'protocol.json';p.write_text('{}');f=tmp_path/'freeze.json';f.write_text('{}')
 with pytest.raises(ValueError):g.guard(p,f,tmp_path/'absent-approval.json','wrong','wrong')
