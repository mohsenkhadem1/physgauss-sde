import json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_static_graded_refinement as m


def test_only_two_new_resolutions_and_original_gates():
    assert m.MESHES==((32,16,16),(40,20,20))
    assert len(m.cases())==2
    assert m.GATES['conditional_actual_h_spatial_indicator_m']==20e-6
    assert 'experiments/run_campaign_c38_static_graded_pilot.py' in m.source_closure()


def test_reuse_checks_hash_before_capture_and_exact_probes(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    points=np.arange(189).reshape(63,3)/1000
    np.savez(tmp_path/'old24.npz',material_probes_m=points,probe_positions_m=points+1)
    capture={'path':'old24.npz','sha256':m.sha(tmp_path/'old24.npz')}
    (tmp_path/'case.json').write_text(json.dumps({'meshes':[{'cells':[24,12,12],'capture':capture}],'coarse_CPU_GPU':{'passed':True}}))
    p={'reference_case_results':{'case':{'path':'case.json','sha256':m.sha(tmp_path/'case.json')}},'reference_mesh24_captures':{'case':capture}}
    row,Q,coarse=m.reused_reference(p,'case',points)
    np.testing.assert_array_equal(Q,points+1);assert row['cells']==[24,12,12] and coarse['passed']
    with pytest.raises(ValueError,match='Original63'):m.reused_reference(p,'case',points+.001)
    (tmp_path/'old24.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed before read'):m.reused_reference(p,'case',points)


def test_frozen_metadata_preparation_without_capture_array_reads(tmp_path,monkeypatch):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'2')
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No array reads during preparation')))
    protocol=tmp_path/'protocol.json';out=m.ROOT/'runs/operator-dev/c38-supported-grading-v01/unlaunched-refinement-test-output'
    digest=m.prepare(protocol,out);p=m.validate(protocol,digest)
    assert p['resources']['global_wall_cap_s']==1800 and p['resources']['per_case_wall_cap_s']==900
    assert len(p['reference_mesh24_captures'])==2 and not out.exists()
    assert p['domain_or_image_cohort_admitted'] is False
    p['reference_mesh24_captures'].pop(next(iter(p['reference_mesh24_captures'])))
    m.write(protocol,p)
    with pytest.raises(ValueError,match='Exact completed24'):m.validate(protocol,m.sha(protocol))
