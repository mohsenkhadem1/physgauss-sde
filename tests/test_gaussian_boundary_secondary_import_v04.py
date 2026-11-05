import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.import_rigid_contact_secondary import copy_verified,file_manifest,import_all
from scripts.gaussian_boundary_flow_v02.object_score import sha


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();np.savez(source/'prediction.npz',x=np.zeros(3))
    (source/'plan.json').write_text('{}')
    (source/'inference-sealed.json').write_text(json.dumps(dict(plan_sha256=sha(source/'plan.json'),forecast_target_access=False,train_end=2)))
    (source/'complete.json').write_text(json.dumps(dict(completed=True,held_target_access=False,fit_end=2,prediction_sha256=sha(source/'prediction.npz'))))
    return source


def test_byte_identical_copy_and_original_preserved(tmp_path):
    source=fixture(tmp_path);before=file_manifest(source);destination=tmp_path/'copy'
    assert copy_verified(source,destination)==before
    assert file_manifest(source)==file_manifest(destination)==before
    assert copy_verified(source,destination)==before
    (destination/'extra').write_text('changed')
    with pytest.raises(ValueError,match='differs'):copy_verified(source,destination)


def test_live_secondary_cohort_rejected_before_copy(tmp_path):
    source=tmp_path/'root';source.mkdir();(source/'summary.json').write_text(json.dumps(dict(all12_terminal=False)))
    addendum=tmp_path/'addendum.json';addendum.write_text('{}')
    with pytest.raises(PermissionError,match='terminal'):import_all(source,tmp_path/'cohort',addendum)
    assert not (tmp_path/'cohort').exists()
