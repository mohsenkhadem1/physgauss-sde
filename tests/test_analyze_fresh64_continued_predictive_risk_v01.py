import json
import types
from copy import deepcopy
import numpy as np
import pytest
from experiments import analyze_fresh64_continued_predictive_risk_v01 as analyze
from tests.test_analyze_fresh64_predictive_risk_v01 import manufactured_object


def mixed_object(tmp_path,monkeypatch):
    row,manifest,directory=manufactured_object(tmp_path)
    monkeypatch.setattr(analyze,'ROOT',tmp_path)
    row['posterior_lineage']={m:{} for m in analyze.MODELS}
    for m in analyze.MODELS:
        for p in analyze.PHASES:
            path=directory/(m+'-'+p+'-posterior-LOCAL.npz')
            row['posterior_lineage'][m][p]={'accepted':dict(source='original_five_levels',
                raw=dict(path=str(path.relative_to(tmp_path)),sha256=manifest[str(path)]))}
    # One archive deliberately lives outside the original object directory.
    replacement=tmp_path/'new-fit-LOCAL.npz'
    original=directory/'nonlinear-normal-r0-posterior-LOCAL.npz'
    replacement.write_bytes(original.read_bytes());manifest[str(replacement)]=analyze.sha(replacement)
    row['posterior_lineage']['nonlinear']['normal-r0']['accepted']=dict(source='refined_axis_extension',
        raw=dict(path=replacement.name,sha256=analyze.sha(replacement)))
    return row,manifest


def test_mixed_archive_paths_preserve_original_statistics(tmp_path,monkeypatch):
    row,manifest=mixed_object(tmp_path,monkeypatch)
    expected=analyze.original.object_inputs(row,manifest,tmp_path)
    actual=analyze.object_inputs(row,manifest,tmp_path)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_old_same_named_archive_cannot_replace_accepted_refinement(tmp_path,monkeypatch):
    row,manifest=mixed_object(tmp_path,monkeypatch)
    replacement=tmp_path/'new-fit-LOCAL.npz';replacement.write_bytes(b'changed')
    with pytest.raises(ValueError,match='accepted posterior archive binding'):
        analyze.object_inputs(row,manifest,tmp_path)


@pytest.mark.parametrize('bad',['missing','unqualified','unknown'])
def test_unaccepted_entries_cannot_be_filtered(tmp_path,monkeypatch,bad):
    row,manifest=mixed_object(tmp_path,monkeypatch)
    if bad=='missing':row['posterior_lineage']['linear']['prefix']['accepted']=None
    elif bad=='unqualified':row['posteriors']['linear']['prefix']['qualified']=False
    else:row['posterior_lineage']['linear']['prefix']['accepted']['source']='relabeled512'
    with pytest.raises(ValueError):analyze.object_inputs(row,manifest,tmp_path)


def test_incomplete_audit_stops_before_terminal_or_npz(tmp_path,monkeypatch):
    monkeypatch.setattr(analyze.os,'sched_getaffinity',lambda _:{17})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(k,'1')
    local=tmp_path/'risk';local.mkdir();auditdir=tmp_path/'audit';auditdir.mkdir()
    old_recipe=tmp_path/'old.json';old_recipe.write_text('{}')
    monkeypatch.setattr(analyze,'ROOT',tmp_path);monkeypatch.setattr(analyze,'LOCAL',local)
    monkeypatch.setattr(analyze,'ORIGINAL_RECIPE_SHA',analyze.sha(old_recipe))
    monkeypatch.setattr(analyze,'sources',lambda:{})
    independent=types.SimpleNamespace(LOCAL=auditdir,sources=lambda:{},
        terminal_metadata=lambda *a:pytest.fail('Audit gate before terminal/raw reads'))
    monkeypatch.setattr(analyze,'auditor',lambda:independent)
    monkeypatch.setattr(analyze.np,'load',lambda *a,**k:pytest.fail('No NPZ before audit'))
    recipe=dict(source_SHA256={},original_recipe=dict(path='old.json'),
        continuation_source_SHA256={},audit_source_SHA256={})
    recipepath=local/'recipe.json';recipepath.write_text(json.dumps(recipe))
    auditpath=auditdir/'result.json';auditpath.write_text(json.dumps(dict(integrity_passed=False)))
    with pytest.raises(ValueError,match='complete independent continued audit'):
        analyze.run(analyze.sha(recipepath),analyze.sha(auditpath))
