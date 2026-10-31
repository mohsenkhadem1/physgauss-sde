import json
import pytest
from scripts.benchmark_improvement_v14 import provenance_b as m


def test_phase_b_regression_pin_and_separate_destination(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE',tmp_path/'runs/new/robust');monkeypatch.setattr(m,'REPORT',tmp_path/'reports/new');m.REPORT.mkdir(parents=True)
    case=dict(name='old',cohort='regression',dataset='SoftVTBench',spec=dict(episode=0))
    p=m.REPORT/'protocol-b.json';p.write_text(json.dumps(dict(regression_cases=[case],fresh_softvt_cases=[])))
    source=tmp_path/'phase-a-forecast.npz';source.write_bytes(b'prior')
    (m.REPORT/'inventory-b.json').write_text(json.dumps(dict(protocol_sha256=m.sha(p),regression={'old':dict(case=case,sha256={'phase-a-forecast.npz':m.sha(source)})})))
    sources=m.snapshot_sources(case)
    out=m.BASE/'evaluation/regression/old';out.mkdir(parents=True);(out/'predictions.npz').write_bytes(b'b');(out/'metadata.json').write_text('{}')
    record=dict(case=case,sources=sources,prediction_sha256=m.sha(out/'predictions.npz'),metadata_sha256=m.sha(out/'metadata.json'))
    assert m.validate_record(case,record,out)
    with pytest.raises(ValueError,match='identity'):m.validate_record(case,record,tmp_path/'runs/new/evaluation/regression/old')
    source.write_bytes(b'changed prior')
    with pytest.raises(ValueError,match='changed'):m.snapshot_sources(case)


def test_matched_frames_do_not_compare_different_support():
    from scripts.benchmark_improvement_v14.robust_evaluation import matched_frames
    def good(frame,error):return dict(frame=frame,coordinate_rmse_mm=error,aligned_shape_rmse_mm=error/2)
    a=dict(frames=[good(1,1),good(2,100),dict(frame=3,failed=True)])
    b=dict(frames=[good(1,2),dict(frame=2,failed=True),good(3,200)])
    result=matched_frames(a,b)
    assert result['frames']==[1] and result['count']==1
    assert result['proposal']['coordinate_rmse_mm']==1 and result['baseline']['coordinate_rmse_mm']==2
    assert matched_frames(dict(frames=[]),b)['proposal']=={}
