import json,os,subprocess
import numpy as np
import pytest
from experiments import validate_surface_prefix_atlas_v01 as validation


def test_all113_synthetic_archives_use_unchanged_pure_checks(monkeypatch,tmp_path):
    design=json.loads(validation.prefix.DESIGN.read_text());design['observed_likelihood_control']['noise_seeds']=[9021,9022,9023,9024]
    dp=tmp_path/'design.json';validation.write(dp,design);monkeypatch.setattr(validation.prefix,'DESIGN',dp)
    points=validation.prefix.point_design(design);rows={}
    for i,u in enumerate(points):
        delta=np.full(24,2e-6*(u[0]**2+.3*u[1]));path=tmp_path/f'{i}.npz'
        np.savez_compressed(path,unit=u,delta_m=delta,linear_response_m=np.zeros(24),response_m=delta)
        rows[i]=dict(completed=True,passed=True,raw=dict(path=str(path),sha256=validation.sha(path)))
    monkeypatch.setattr(validation.prefix,'solve_point',lambda *a,**k:pytest.fail('No PDE permitted'))
    check,atlas=validation.evaluate_rows(dict(points_unit_LOCAL=points.tolist()),rows,'synthetic')
    assert check['passed']and check['all32_queries_retained']and check['all12_likelihood_cases_retained']
    assert atlas.delta.shape==(9,9,1,24)


def test_corrupt_field_refused_before_numeric_read(monkeypatch,tmp_path):
    path=tmp_path/'raw.npz';path.write_bytes(b'changed')
    rows={i:dict(completed=True,passed=True,raw=dict(path=str(path),sha256='wrong'))for i in range(113)}
    monkeypatch.setattr(validation.np,'load',lambda *a,**k:pytest.fail('No numerical read before SHA validation'))
    with pytest.raises(ValueError,match='hash-bound'):validation.evaluate_rows(dict(points_unit_LOCAL=np.zeros((113,2))),rows,'synthetic')


def test_missing_release_no_validation_calls(monkeypatch,tmp_path):
    monkeypatch.setattr(validation,'LOCAL',tmp_path);monkeypatch.setattr(validation,'guard',lambda:None)
    p=dict(source_SHA256={},origin_SHA256={},resources=validation.LIMITS)
    validation.write(tmp_path/'protocol.json',p);digest=validation.sha(tmp_path/'protocol.json')
    monkeypatch.setattr(validation,'sources',lambda:{});monkeypatch.setattr(validation,'inputs',lambda:({}, {}, {}))
    monkeypatch.setattr(validation,'evaluate_rows',lambda *a:pytest.fail('No released execution'))
    with pytest.raises(FileNotFoundError):validation.run(digest)


def test_cpu17_no_gpu_guard():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    result=subprocess.run(['taskset','-c','17',validation.os.sys.executable,'-m','experiments.validate_surface_prefix_atlas_v01','guard'],cwd=validation.ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
