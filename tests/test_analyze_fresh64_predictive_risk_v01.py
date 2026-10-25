import json
from pathlib import Path
import numpy as np
import pytest
from experiments import analyze_fresh64_predictive_risk_v01 as analyze


def manufactured_object(tmp_path):
    directory = tmp_path/'object-00'; directory.mkdir()
    manifest, targets = {}, []
    def archive(name, **arrays):
        path = directory/name; np.savez_compressed(path, **arrays)
        manifest[str(path)] = analyze.sha(path)
        return path
    for i, label in enumerate(analyze.fresh.study.TARGETS):
        response = (i+1)*1e-4 + np.arange(24)*1e-7
        targets.append(response); archive(label+'-truth-LOCAL.npz', response_m=response)
    target = np.asarray(targets).ravel()
    row = dict(index=0, posteriors={model:{} for model in analyze.MODELS})
    for mi, model in enumerate(analyze.MODELS):
        for pi, phase in enumerate(analyze.PHASES):
            offset = (mi+1)*(pi+1)*1e-6
            p = archive(model+'-'+phase+'-posterior-LOCAL.npz', mean=target+offset, SD=np.full(96,2e-6))
            witness = {axis:dict(actual_absolute_errors_m=dict(predictive_mean=1e-8 if axis=='inner' else 2e-8,
                        predictive_SD=3e-8 if axis=='inner' else 2e-8)) for axis in ('inner','outer')}
            row['posteriors'][model][phase] = dict(qualified=True,physical_valid=True,
                raw_SHA256=analyze.sha(p),strict_absolute_checks=dict(final_axis_witnesses=witness))
    return row, manifest, directory


def test_clean_target_squared_error_variance_and_achieved_refinement(tmp_path):
    row, manifest, _ = manufactured_object(tmp_path)
    mse, variance, refinement = analyze.object_inputs(row, manifest, tmp_path)
    np.testing.assert_allclose(mse, (np.arange(1,3)[:,None]*np.arange(1,4)*1e-6)**2, rtol=1e-12)
    np.testing.assert_allclose(variance, 4e-12, rtol=1e-15)
    np.testing.assert_allclose(refinement[:,:,0], 2e-8)
    np.testing.assert_allclose(refinement[:,:,1], 3e-8)


def test_unqualified_or_changed_archive_never_silently_filtered(tmp_path):
    row, manifest, directory = manufactured_object(tmp_path)
    row['posteriors']['linear']['prefix']['qualified'] = False
    with pytest.raises(ValueError,match='No filtering'): analyze.object_inputs(row,manifest,tmp_path)
    row['posteriors']['linear']['prefix']['qualified'] = True
    path = directory/(analyze.fresh.study.TARGETS[0]+'-truth-LOCAL.npz')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='archive bytes'): analyze.object_inputs(row,manifest,tmp_path)


def test_negative_posterior_SD_rejected_even_with_matching_hashes(tmp_path):
    row, manifest, directory = manufactured_object(tmp_path)
    path = directory/'linear-prefix-posterior-LOCAL.npz'
    np.savez_compressed(path,mean=np.zeros(96),SD=np.full(96,-1e-6))
    manifest[str(path)] = row['posteriors']['linear']['prefix']['raw_SHA256'] = analyze.sha(path)
    with pytest.raises(ValueError,match='nonnegative SD'): analyze.object_inputs(row,manifest,tmp_path)


def test_incomplete_primary_audit_blocks_any_raw_archive_read(monkeypatch,tmp_path):
    monkeypatch.setattr(analyze.os,'sched_getaffinity',lambda _:{17})
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): monkeypatch.setenv(key,'1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    local = tmp_path/'analysis'; local.mkdir(); auditdir = tmp_path/'audit'; auditdir.mkdir()
    monkeypatch.setattr(analyze,'LOCAL',local); monkeypatch.setattr(analyze.independent,'LOCAL',auditdir)
    monkeypatch.setattr(analyze,'sources',lambda:{})
    recipe = local/'recipe.json'
    recipe.write_text(json.dumps(dict(source_SHA256={},primary_source_SHA256={},audit_source_SHA256={})))
    audit = auditdir/'result.json'; audit.write_text(json.dumps(dict(integrity_passed=False)))
    monkeypatch.setattr(analyze.np,'load',lambda *a,**k:pytest.fail('No data before complete audit'))
    monkeypatch.setattr(analyze.independent,'terminal_metadata',lambda *a,**k:pytest.fail('Incomplete audit refused first'))
    with pytest.raises(ValueError,match='complete independent'): analyze.run(analyze.sha(recipe),analyze.sha(audit))
