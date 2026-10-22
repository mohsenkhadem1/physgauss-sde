"""Temporary metadata only; old full case JSON is neither needed nor opened."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_static_graded_refinement_v03 as m


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_closure',lambda:['source.py']);(tmp_path/'source.py').write_text('# synthetic\n')
    def put(name,obj):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return {'path':name,'sha256':m.sha(p)}
    prior=[];derived=[];integrity=[]
    for i,case in enumerate(m.cases()):
        meshes=[]
        for n in (12,20,24):
            cap=tmp_path/f'capture-{i}-{n}.npz';cap.write_bytes(b'no array reads during metadata preparation'+bytes([i,n]));record={'path':cap.name,'sha256':m.sha(cap)};cells=[n,n//2,n//2]
            meshes.append({'cells':cells,'capture':record,'hmax_m':.3/n,'seconds':n*n/10})
            derived.append({'case':case,'cells':cells,'passed':True,'capture':record,'hmax_m':.3/n,'revalidated_physics_passed':True,'diagnostics':{},'R24_N':1e-12,'R80_N':1e-12,'whole_J':{'all_elements_positive':True,'minimum_lower_bound':.9}})
            integrity.append({'case':case['id'],'cells':cells,'capture':record})
        prior.append({'case':case,'meshes':meshes})
    values={'pilot_protocol':{'resources':{},'mapping':m.MAPPING,'gates':m.GATES,'domain_or_image_cohort_admitted':False},'preincident_compact_summary':{'completed':True,'spatial_cases_failed':2,'independent_coarse_controls_passed':2,'rows':prior},'preincident_terminal_proof':{'completed':True},'raw_integrity':{'all6_capture_hashes_match_preincident_completed_summary':True,'records':integrity},'incident':{'original_full_case_JSON_recovered':False},'revalidation_protocol':{'frozen':True},'revalidation_summary':{'completed':True,'passed':True,'rows':derived},'revalidation_terminal_proof':{'all_source_pins_unchanged':True,'worker_PID_absent':True}}
    for key,name in m.METADATA_INPUTS.items():put(name,values[key])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'2')


def test_destination_only_exclusive_write_no_case_JSON_or_array_reads(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()};destination=tmp_path/'new.json';original_open=Path.open
    def guard(path,mode='r',*args,**kw):
        if any(x in mode for x in 'wax+'):assert path==destination and mode=='x'
        if path.name=='result.json':raise AssertionError('Missing old full caseJSON must NEVER be opened')
        return original_open(path,mode,*args,**kw)
    monkeypatch.setattr(Path,'open',guard);monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No arrays during preparation')))
    digest=m.prepare(destination,tmp_path/'new-output');p=m.validate(destination,digest)
    assert p['meshes']==[[32,16,16],[40,20,20]] and len(p['reference_mesh24'])==2
    assert p['original_full_case_JSON_recovered'] is False
    assert p['resources']['global_wall_cap_s']==1800 and p['resources']['per_case_wall_cap_s']==900
    assert all(path.read_bytes()==value for path,value in before.items())
    with pytest.raises(ValueError):m.prepare(destination,tmp_path/'other')


def test_different_capture_binding_rejected_before_arrays(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    p=tmp_path/m.METADATA_INPUTS['revalidation_summary'];d=json.loads(p.read_text());d['rows'][2]['capture']=d['rows'][1]['capture'];p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='capture binding'):m.references()


def test_new_derived_summary_and_capture_changes_invalidate_protocol(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);p=tmp_path/'new.json';digest=m.prepare(p,tmp_path/'new-out');m.validate(p,digest)
    raw=tmp_path/'capture-0-24.npz';raw.write_bytes(b'changed')
    with pytest.raises(ValueError,match='capture binding'):m.validate(p,digest)


def test_same_points_and_no_warmstart_or_repeated24_scope():
    assert m.MESHES==((32,16,16),(40,20,20)) and len(m.cases())==2
    assert m.GATES['conditional_actual_h_spatial_indicator_m']==20e-6
