import json
from pathlib import Path
import numpy as np
import pytest
from experiments import revalidate_c38_graded_captures as m


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'sources',lambda:['source.py']);(tmp_path/'source.py').write_text('# synthetic\n')
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return {'path':name,'sha256':m.sha(p)}
    summary=put(m.SUMMARY,{'completed':True});records=[]
    for i in range(6):
        r={k:put(f'raw/{i}-{k}.json',{'synthetic':True}) for k in ('capture','latest_pointer','latest_metadata','latest_data')};records.append(r)
    put(m.INTEGRITY,{'completed':True,'all6_capture_hashes_match_preincident_completed_summary':True,'records':records})
    put('runs/operator-dev/c38-static-graded-refinement-v01/failed-preparation-overwrite-bytes.json',{'incident':True})


def test_prepare_exclusive_exact_destination_no_arrays_or_other_writes(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()};destination=tmp_path/'new.json';original_open=Path.open
    def guard(path,mode='r',*args,**kw):
        if any(x in mode for x in 'wax+'):assert path==destination and mode=='x'
        return original_open(path,mode,*args,**kw)
    monkeypatch.setattr(Path,'open',guard);monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No arrays/GPU in metadata preparation')))
    digest=m.prepare(destination,tmp_path/'output');p=m.validate(destination,digest)
    assert p['original_case_JSON_recovered'] is False and p['resources']['wall_cap_s']==180
    assert all(path.read_bytes()==value for path,value in before.items()) and not (tmp_path/'output').exists()
    with pytest.raises(ValueError):m.prepare(destination,tmp_path/'output')


def test_changed_capture_rejects_before_evaluation(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);p=tmp_path/'new.json';digest=m.prepare(p,tmp_path/'output')
    (tmp_path/'raw/0-capture.json').write_text('changed')
    with pytest.raises(ValueError,match='raw artifact changed'):m.validate(p,digest)
