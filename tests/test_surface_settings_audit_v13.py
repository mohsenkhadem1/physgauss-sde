import hashlib,json
from pathlib import Path
import numpy as np
from scripts.coupled_contact_v13 import surface_verify as v

class Audit:
    def __init__(self):self.errors=[]
    def check(self,value,label):
        if not value:self.errors.append(label)
    def close(self,a,b,label):self.check(np.isclose(a,b,rtol=0,atol=1e-15),label)

def test_independent_override_halfstep_and_cache_hash(tmp_path,monkeypatch):
    base=tmp_path/'runs';report=tmp_path/'reports';report.mkdir()
    monkeypatch.setattr(v,'BASE',base);monkeypatch.setattr(v,'REPORT',report)
    p=dict(reference_cells=[8]*3,coarse_cells=[4]*3,dt_reference=25e-6,dt_coarse=12.5e-6,duration_seconds=.6,surface_spacing_m=.0015,friction_quadrature_level=1,reference_settings_by_scenario={'edge_drop':dict(cells=[12]*3,dt=12.5e-6)})
    (report/'protocol.json').write_text(json.dumps(p))
    path=base/'reference-refinement/o/edge_drop';path.mkdir(parents=True)
    status=dict(success=False,job=dict(path=str(path.relative_to(base)),scenario='edge_drop',family='fem',reference=True,refinement=True),protocol_sha256=hashlib.sha256((report/'protocol.json').read_bytes()).hexdigest(),simulation_settings=dict(cells=[12]*3,dt=6.25e-6,duration_seconds=.6,scene_source='engineered_scenarios.py',mass_operator='consistent-P1',contact_formulation='clipped-normal-refined-friction-sparse-sphere',surface_spacing_m=.0015,friction_quadrature_level=1))
    f=path/'complete.json';f.write_text(json.dumps(status));a=Audit();assert v.verify_settings(a,p)==1;assert not a.errors
    status['simulation_settings']['dt']=12.5e-6;status['protocol_sha256']='stale';f.write_text(json.dumps(status));a=Audit();v.verify_settings(a,p)
    assert any('settings' in e for e in a.errors) and any('protocol hash' in e for e in a.errors)

def test_non_nested_override_rejected(tmp_path,monkeypatch):
    base=tmp_path/'runs';report=tmp_path/'reports';report.mkdir();monkeypatch.setattr(v,'BASE',base);monkeypatch.setattr(v,'REPORT',report)
    p=dict(reference_cells=[10]*3,coarse_cells=[4]*3,dt_reference=.000025,dt_coarse=.0000125,duration_seconds=.6,surface_spacing_m=.0015,friction_quadrature_level=1)
    (report/'protocol.json').write_text(json.dumps(p));path=base/'reference/o/squeeze';path.mkdir(parents=True)
    status=dict(success=False,job=dict(path=str(path.relative_to(base)),scenario='squeeze',family='fem',reference=True),protocol_sha256=hashlib.sha256((report/'protocol.json').read_bytes()).hexdigest(),simulation_settings={})
    (path/'complete.json').write_text(json.dumps(status));a=Audit();v.verify_settings(a,p);assert any('nested' in e for e in a.errors)
