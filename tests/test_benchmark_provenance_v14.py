import json
from pathlib import Path
import pytest
from scripts.benchmark_improvement_v14 import provenance as m


def test_pinned_historical_input_and_complete_record_mutations(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE',tmp_path/'runs/new');monkeypatch.setattr(m,'REPORT',tmp_path/'reports/new')
    m.REPORT.mkdir(parents=True)
    case=dict(name='old',cohort='regression',dataset='SoftVTBench',spec=dict(episode=0))
    p=m.REPORT/'protocol.json';p.write_text(json.dumps(dict(historical_cases=[case],fresh_softvt_cases=[])))
    inp=tmp_path/'old-input.npz';inp.write_bytes(b'prefix');inv=dict(protocol_sha256=m.sha(p),cases={'old':dict(case=case,sha256={'old-input.npz':m.sha(inp)})})
    (m.REPORT/'input-inventory.json').write_text(json.dumps(inv))
    out=m.BASE/'evaluation/regression/old';out.mkdir(parents=True);(out/'predictions.npz').write_bytes(b'pred');(out/'metadata.json').write_text('{}')
    record=dict(case=case,sources=m.snapshot_sources(case),prediction_sha256=m.sha(out/'predictions.npz'),metadata_sha256=m.sha(out/'metadata.json'))
    assert m.validate_record(case,record,out)
    with pytest.raises(ValueError,match='identity'):m.snapshot_sources(dict(case,cohort='development'))
    (out/'metadata.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='artifact'):m.validate_record(case,record,out)
    inp.write_bytes(b'future-contamination')
    with pytest.raises(ValueError,match='Historical input'):m.snapshot_sources(case)


def test_wrong_record_case_rejected_before_input_access(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'declared',lambda c:c);monkeypatch.setattr(m,'BASE',tmp_path)
    case=dict(name='x',cohort='c')
    with pytest.raises(ValueError,match='identity'):m.validate_record(case,dict(case=dict(name='y')),tmp_path/'evaluation/c/x')
