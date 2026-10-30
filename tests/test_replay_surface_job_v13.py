import json
from pathlib import Path
import pytest
from scripts.coupled_contact_v13 import replay_surface_job as r

def protocol():
 return {'fem_young_grid':[2000,3000],'native_scale_grid':[1e5], 'friction_grid':[0,.2], 'objects':[{'id':'dev0','young':6000,'friction':.35}], 'forecast_scenarios':['slide']}

def test_exact_declared_library_and_reference_variants():
 p=protocol();jobs=r.declared(p)
 assert len(jobs)==14
 assert len({j['path'] for j in jobs})==len(jobs)
 assert r.select_job('library/fem/1-1',p)[0]['scale']==3000
 assert r.select_job('reference-refinement/dev0/slide',p)[0]['refinement']
 assert r.select_job('reference-spatial/dev0/slide',p)[0]['spatial']
 for name in ('library/fem/99-0','../library/fem/0-0','prediction/missing/slide/fem','prediction/dev0/squeeze/fem'):
  with pytest.raises(ValueError):r.select_job(name,p)

def test_fresh_output_rejects_existing_symlink_and_protected(tmp_path):
 assert r.fresh_output(tmp_path/'fresh')==tmp_path/'fresh'
 with pytest.raises(ValueError):r.fresh_output(tmp_path)
 (tmp_path/'link').symlink_to(tmp_path/'absent')
 with pytest.raises(ValueError):r.fresh_output(tmp_path/'link')
 with pytest.raises(ValueError):r.fresh_output(r.BASE/'never-created-replay-test')
 with pytest.raises(ValueError):r.fresh_output(r.ROOT/'scripts'/'never-created-replay-test')

def test_selected_prediction_requires_saved_matching_metadata(tmp_path,monkeypatch):
 p=protocol();monkeypatch.setattr(r,'BASE',tmp_path);monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'REPORT',tmp_path/'report')
 (tmp_path/'report').mkdir();(tmp_path/'report/protocol.json').write_text(json.dumps(p))
 selection=tmp_path/'inference/dev0';selection.mkdir(parents=True)
 (selection/'selection.json').write_text(json.dumps({'fem':{'success':True,'scale':3000,'friction':.2}}))
 from scripts.coupled_contact_v13 import surface_pipeline
 monkeypatch.setattr(surface_pipeline,'simulation_settings',lambda job,p:{'dt':1e-5})
 path='prediction/dev0/slide/fem';folder=tmp_path/path;folder.mkdir(parents=True)
 job={'path':path,'family':'fem','scenario':'slide','scale':3000,'friction':.2,'reference':False}
 status={'job':job,'simulation_settings':{'dt':1e-5},'protocol_sha256':r.sha(tmp_path/'report/protocol.json')}
 (folder/'complete.json').write_text(json.dumps(status))
 actual,evidence=r.select_job(path,p);assert actual==job and len(evidence)==2
 status['job']['friction']=.1;(folder/'complete.json').write_text(json.dumps(status))
 with pytest.raises(ValueError):r.select_job(path,p)
