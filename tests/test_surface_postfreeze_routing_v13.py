import json,hashlib
import numpy as np
from scripts.coupled_contact_v13 import surface_verify_scoped as scoped
from scripts.coupled_contact_v13 import surface_finish_scoped, surface_finish

class Audit:
    def __init__(self):self.errors=[]
    def check(self,condition,label):
        if not condition:self.errors.append(label)
    def close(self,a,b,label):self.check(np.isclose(a,b),label)

def test_scoped_audit_retains_failed_production_and_skips_legacy_and_supervisor(tmp_path):
    m=scoped.scoped_module();m.BASE=tmp_path/'runs';m.REPORT=tmp_path/'reports';m.REPORT.mkdir()
    p=dict(coarse_cells=[4]*3,reference_cells=[8]*3,dt_coarse=12.5e-6,dt_reference=25e-6,duration_seconds=.6,surface_spacing_m=.0015,friction_quadrature_level=1,reference_settings_by_scenario={'slide':dict(cells=[8]*3,companion_cells=[10]*3,dt=25e-6)})
    (m.REPORT/'protocol.json').write_text(json.dumps(p));sha=hashlib.sha256((m.REPORT/'protocol.json').read_bytes()).hexdigest()
    for directory in scoped.PRODUCTION_ROOTS:
        path=m.BASE/directory/'o'/'slide';path.mkdir(parents=True);ref=directory.startswith('reference');spatial=directory=='reference-spatial';half=directory=='reference-refinement'
        job=dict(path=str(path.relative_to(m.BASE)),family='fem',scenario='slide',reference=ref)
        if spatial:job['spatial']=True
        if half:job['refinement']=True
        settings=dict(cells=[10 if spatial else 8 if ref else 4]*3,dt=12.5e-6 if half or not ref else 25e-6,duration_seconds=.6,scene_source='engineered_scenarios.py',mass_operator='consistent-P1',contact_formulation='clipped-normal-refined-friction-sparse-sphere',surface_spacing_m=.0015,friction_quadrature_level=1)
        (path/'complete.json').write_text(json.dumps(dict(job=job,success=False,simulation_settings=settings,protocol_sha256=sha)))
    bad=m.BASE/'parameter-range-development/case';bad.mkdir(parents=True);(bad/'complete.json').write_text(json.dumps({'job':['legacy']}));(m.BASE/'complete.json').write_text(json.dumps({'status':'supervisor'}))
    a=Audit();assert m.verify_settings(a,p)==5 and not a.errors
    f=m.BASE/'prediction/o/slide/complete.json';d=json.loads(f.read_text());d['protocol_sha256']='wrong';f.write_text(json.dumps(d));a=Audit();assert m.verify_settings(a,p)==5;assert any('protocol hash' in x for x in a.errors)

def test_finish_changes_only_verifier_and_retains_original_deadline_globals():
    before=list(surface_finish.STAGES);m=surface_finish_scoped.implementation()
    assert m.STAGES==[(('surface_verify_scoped' if name=='surface_verify' else name),args) for name,args in before]
    assert surface_finish.STAGES==before
    assert m.remaining is surface_finish.remaining and m.BASE==surface_finish.BASE
