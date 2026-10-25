import json,os,subprocess
import numpy as np
import pytest
from experiments import run_surface_prefix_atlas_recovery_v01 as recovery


class Child:
    pid=123
    def __init__(self,exitcode):self.exitcode=exitcode
    def poll(self):return self.exitcode


def test_proc_exit_race_allows_only_reaped_or_zombie():
    zombie=lambda:'Name:\tpython\nState:\tZ (zombie)\n'
    missing=lambda:(_ for _ in ()).throw(FileNotFoundError())
    assert recovery.owned_RSS(Child(0),read_status=missing)==0
    assert recovery.owned_RSS(Child(0),read_status=lambda:'State:\tR\n')==0
    assert recovery.owned_RSS(Child(None),read_status=zombie)==0
    with pytest.raises(RuntimeError):recovery.owned_RSS(Child(None),read_status=missing)
    with pytest.raises(RuntimeError):recovery.owned_RSS(Child(None),read_status=lambda:'State:\tR (running)\n')
    assert recovery.owned_RSS(Child(None),read_status=lambda:'State:\tR\nVmRSS:\t12345 kB\n')==12345*1024


def test_recovery_has_only_five_new_points_and_108_reused():
    assert recovery.MISSING==[104,105,106,108,112]
    assert len(set(range(113))-set(recovery.MISSING))==108
    assert recovery.LIMITS['worker_CPU']==17 and recovery.LIMITS['controller_CPU']==16


def test_generated_guards_no_gpu():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for cmd in(recovery.worker_command('not-a-protocol',guard_only=True),['taskset','-c','16',recovery.sys.executable,'-m','experiments.run_surface_prefix_atlas_recovery_v01','controller-guard']):
        result=subprocess.run(cmd,cwd=recovery.ROOT,env=env,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr


def test_missing_release_prevents_process_launch(monkeypatch,tmp_path):
    monkeypatch.setattr(recovery,'LOCAL',tmp_path);monkeypatch.setattr(recovery,'guard',lambda _:None);monkeypatch.setattr(recovery,'validate',lambda _:(dict(source_SHA256={}),{},{}))
    monkeypatch.setattr(recovery.subprocess,'Popen',lambda *a,**k:pytest.fail('No actual processes'))
    with pytest.raises(FileNotFoundError):recovery.run('toy')


def test_any_new_failed_point_retained_without_old_field_reads(monkeypatch,tmp_path):
    monkeypatch.setattr(recovery,'LOCAL',tmp_path)
    rows={i:dict(index=i,passed=True)for i in range(113)if i not in recovery.MISSING}
    oldp=dict(points_unit_LOCAL=np.zeros((113,2)).tolist())
    monkeypatch.setattr(recovery,'validate',lambda _:(dict(),oldp,rows))
    points=[dict(index=i,completed=i!=104,passed=i!=104,unit_SHA256=recovery.array_sha([0.,0.]),protocol_SHA256='toy',original_protocol_SHA256=recovery.OLD_PROTOCOL_SHA)for i in recovery.MISSING]
    directory=tmp_path/'new-points-LOCAL';directory.mkdir()
    for row in points:recovery.write(directory/f'point-{row["index"]:03d}.json',row)
    recovery.write(tmp_path/'worker-result-LOCAL.json',dict(completed=True,protocol_SHA256='toy',points=points))
    monkeypatch.setattr(recovery.np,'load',lambda *a,**k:pytest.fail('No raw reads before full physical admission'))
    result=recovery.summarize('toy')
    assert result['all113_retained']and result['physical_passed']==112 and not result['qualified']


def test_new_point_metadata_tampering_is_refused_before_raw_reads(monkeypatch,tmp_path):
    monkeypatch.setattr(recovery,'LOCAL',tmp_path)
    rows={i:dict(index=i,passed=True)for i in range(113)if i not in recovery.MISSING}
    oldp=dict(points_unit_LOCAL=np.zeros((113,2)).tolist())
    monkeypatch.setattr(recovery,'validate',lambda _:(dict(),oldp,rows))
    directory=tmp_path/'new-points-LOCAL';directory.mkdir()
    points=[dict(index=i,completed=True,passed=True,unit_SHA256=recovery.array_sha([0.,0.]),protocol_SHA256='toy',original_protocol_SHA256=recovery.OLD_PROTOCOL_SHA)for i in recovery.MISSING]
    for row in points:recovery.write(directory/f'point-{row["index"]:03d}.json',row)
    recovery.write(tmp_path/'worker-result-LOCAL.json',dict(completed=True,protocol_SHA256='toy',points=points))
    changed=dict(points[0],protocol_SHA256='altered')
    recovery.write(directory/'point-104.json',changed)
    monkeypatch.setattr(recovery.np,'load',lambda *a,**k:pytest.fail('No raw reads after tampering'))
    with pytest.raises(ValueError,match='metadata/protocol'):recovery.summarize('toy')
