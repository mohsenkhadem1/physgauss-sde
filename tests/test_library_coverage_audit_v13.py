import hashlib,json
from scripts.coupled_contact_v13.library_coverage_audit import audit


def test_all_declared_failures_remain_and_parameter_tampering_is_detected(tmp_path):
    base=tmp_path/'runs';report=tmp_path/'reports';report.mkdir()
    p=dict(coarse_cells=[4]*3,dt_coarse=12.5e-6,duration_seconds=.6,fps=100,surface_spacing_m=.0015,friction_quadrature_level=1,fem_young_grid=list(range(1,8)),native_scale_grid=list(range(11,18)),friction_grid=[0,.2,.4,.6,.8])
    (report/'protocol.json').write_text(json.dumps(p));sha=hashlib.sha256((report/'protocol.json').read_bytes()).hexdigest()
    (report/'method-freeze.json').write_text(json.dumps(dict(protocol_sha256=sha,source_sha256={})))
    settings=dict(cells=[4]*3,dt=12.5e-6,duration_seconds=.6,scene_source='engineered_scenarios.py',mass_operator='consistent-P1',contact_formulation='clipped-normal-refined-friction-sparse-sphere',surface_spacing_m=.0015,friction_quadrature_level=1)
    for family,scales in [('fem',p['fem_young_grid']),('native',p['native_scale_grid']),('rigid',[6000.])]:
        for i,scale in enumerate(scales):
            for j,mu in enumerate(p['friction_grid']):
                path=base/'library'/family/f'{i}-{j}';path.mkdir(parents=True)
                job=dict(path=f'library/{family}/{i}-{j}',family=family,scenario='squeeze',scale=scale,friction=mu,reference=False)
                (path/'complete.json').write_text(json.dumps(dict(job=job,success=False,error='Step inverted/degenerate element',simulation_settings=settings,protocol_sha256=sha)))
    result=audit(base,report,tmp_path)
    assert result['passed'] and result['present']==75
    assert result['families']['native']['failed']==35 and result['families']['native']['valid_completed']==0
    path=base/'library/native/0-0/complete.json';status=json.loads(path.read_text());status['job']['scale']=999
    path.write_text(json.dumps(status));result=audit(base,report,tmp_path)
    assert not result['passed'] and any('grid job mismatch' in e for e in result['errors'])
