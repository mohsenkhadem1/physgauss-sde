from pathlib import Path
import json
import pytest
from experiments import run_campaign_c38_static_graded_pilot as m
from experiments.campaign_c38_static_family import build_geometry
import numpy as np


def test_fixed_pilot_is_same_physical_worst_and_both_end_control():
    rows=m.cases()
    assert len(rows)==2 and {r['support'] for r in rows}=={'both_ends','separated_bottom_patches'}
    assert all(r['young_Pa']==75000. and r['Fz_N']==-2. for r in rows)
    assert m.MESHES==((12,6,6),(20,10,10),(24,12,12))
    assert m.GATES['conditional_actual_h_spatial_indicator_m']==20e-6
    assert 'experiments/campaign_c38_supported_grading.py' in m.source_closure()
    # The independent sparse coarse control is physically and geometrically the
    # same uniform4x2x2 control, not a silently mismatched CPU mesh.
    a=m.build_geometry((4,2,2),75000.,'separated_bottom_patches',-2.,rule='exact80')
    b=build_geometry((4,2,2),75000.,'separated_bottom_patches',-2.,rule='exact80')
    for name in ('rest','elements','tets','mask','load'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def test_undeclared_case_rejected_before_output_or_GPU(tmp_path):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    with pytest.raises(ValueError,match='Exactly declared'):
        m.worker({'cases':m.cases()},protocol,'undeclared')
    assert sorted(p.name for p in tmp_path.iterdir())==['protocol.json']


def test_frozen_preparation_keeps_uniform_failures_and_rejects_metadata_tamper(tmp_path,monkeypatch):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(name,'2')
    # Preparation only reads completed metadata, never captures or new forces.
    protocol=tmp_path/'protocol.json';out=m.ROOT/'runs/operator-dev/c38-supported-grading-v01/unlaunched-test-output'
    digest=m.prepare(protocol,out)
    p=m.validate(protocol,digest)
    assert p['resources']['global_wall_cap_s']==1200
    assert p['resources']['per_case_wall_cap_s']==600
    assert p['domain_or_image_cohort_admitted'] is False
    assert p['existing_uniform_matrix']['path'].endswith('completed-summary.json')
    assert not out.exists()
    p['mapping']='uniform';m.write(protocol,p)
    with pytest.raises(ValueError):m.validate(protocol,m.sha(protocol))
