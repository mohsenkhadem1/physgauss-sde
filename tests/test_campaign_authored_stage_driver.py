"""No physics: exercise orchestration failure, cap, publication and restart rules."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import run_campaign_authored_stage as driver


CONFIG = Path(__file__).resolve().parents[1]/'configs/gaussian_campaign_actual_force_envelope_v01.json'


def test_complete_initial_matrix_preserves_each_declared_request_without_reading_responses():
    protocol = json.loads(CONFIG.read_text())
    manifest = driver.build_initial_manifest(protocol, protocol_sha256='a'*64,
        source_sha256={'experiments/frozen.py': 'b'*64}, authored_auditor_source_sha256='c'*64,
        output_root='runs/fresh-stage')
    original = {r['id']: r for r in protocol['initial_cell_plan'] if r['stage']=='initial_temporal'}
    assert len(manifest['cells']) == len(original) == 51
    for row in manifest['cells']:
        for key, value in original[row['id']].items(): assert row[key] == value
        assert row['template_id'] == row['id']
        assert row['output'] == 'runs/fresh-stage/'+row['id']
    assert {r['case'] for r in manifest['cells'][:8]} == {'t0000','t0001','t0002','t0003'}
    assert manifest['cells'][0]['id']=='t0001__push_pulse__temporal__hz120'
    assert {r['response'] for r in manifest['cells'][1:4]} == {'zero_external_force'}


@pytest.mark.parametrize('path', ['../outside', '/absolute', '.', 'runs/../../outside', 'runs//ambiguous', 'runs\\ambiguous'])
def test_persisted_paths_cannot_depend_on_machine_or_escape_repository(path):
    with pytest.raises(ValueError): driver.relative_path(path)


def test_publication_requires_actual_fetched_git_blob_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    subprocess.run(['git','init','--quiet',str(tmp_path)],check=True)
    p=tmp_path/'source.txt';p.write_text('frozen source\n')
    subprocess.run(['git','add','source.txt'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','--quiet','-m','fixture'],cwd=tmp_path,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tmp_path,text=True).strip()
    driver.require_published_files(commit,['source.txt'])
    p.write_text('changed after publication\n')
    with pytest.raises(ValueError,match='bytes differ'): driver.require_published_files(commit,['source.txt'])


def fake_stage(tmp_path, monkeypatch, *, fail_first=False, broken_artifact=False, malformed_record=False, wrong_length=False, first_gate=False, audit_passed=True):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    source=tmp_path/'experiments/run_campaign_authored_stage.py';source.parent.mkdir();source.write_text('driver fixture')
    monkeypatch.setattr(driver,'SOURCE',source)
    stage=tmp_path/'runs/stage';stage.mkdir(parents=True)
    cells=[{'id':f'cell{i}','output':f'runs/stage/cell{i}','hz':120} for i in range(3)]
    manifest={'cells':cells};driver.write(stage/'manifest.json',manifest)
    driver.write(stage/'resource.json',{'max_parallel_workers':2,
        'first_cell_independent_audit_before_parallel_launch':first_gate,'first_cell_id':'cell0'})
    seal={'resolved_manifest':{'path':'runs/stage/manifest.json'},
          'resource_declaration':{'path':'runs/stage/resource.json'},
          'stage_driver_source':{'path':source.relative_to(tmp_path).as_posix(),'sha256':driver.digest(source)}}
    driver.write(stage/'seal.json',seal)
    def validate_seal(path,cell_id):
        assert path=='runs/stage/seal.json'  # The real guard accepts canonical relative records only.
        return {},manifest,cells[0],{}
    fake_runner=SimpleNamespace(validate_seal=validate_seal)
    monkeypatch.setitem(sys.modules,'campaign_authored_force_case',fake_runner)
    publications=[];monkeypatch.setattr(driver,'require_published_files',lambda commit,paths:publications.append((commit,paths)))
    launches=[]
    def audit(seal_path,cell_id):
        assert len(launches)==1  # No second physical job starts until this independent audit passes.
        assert cell_id=='cell0' and seal_path=='runs/stage/seal.json'
        return {'id':cell_id,'passed':audit_passed,'status':'synthetic test only'}
    monkeypatch.setattr(driver,'independent_cell_audit',audit)
    class Process:
        def __init__(self,command,**kwargs):
            assert kwargs['cwd']==tmp_path
            assert all(kwargs['env'][name]=='2' for name in driver.THREAD_ENV)
            self.index=len(launches);self.pid=9000+self.index;launches.append(command)
            out=tmp_path/command[-1];out.mkdir()
            driver.write(out/'execution.json',{'terminal':True,'completed':True,'pid':self.pid,
                                              'completed_steps':360,'planned_steps':360})
            (out/'trajectory.npz').write_bytes(b'fake trajectory; no simulation')
            (out/'steps.jsonl').write_text('fake steps; no simulation\n')
            driver.write(out/'summary.json',{'completed':True,'cell_id':command[command.index('--cell-id')+1],
                                           'frames':61,'trajectory_sha256':driver.digest(out/'trajectory.npz'),
                                           'steps_sha256':driver.digest(out/'steps.jsonl')})
            if self.index==0 and broken_artifact:(out/'trajectory.npz').unlink()
            if self.index==0 and malformed_record:driver.write(out/'execution.json',[])
            if self.index==0 and wrong_length:
                driver.write(out/'execution.json',{'terminal':True,'completed':True,'pid':self.pid,
                                                  'completed_steps':30,'planned_steps':30})
        def poll(self):return 1 if fail_first and self.index==0 else 0
    monkeypatch.setattr(driver.subprocess,'Popen',Process)
    monkeypatch.setattr(driver.time,'sleep',lambda _:None)
    return stage,launches,publications


@pytest.mark.parametrize('failure', ['nonzero_exit','missing_artifact','malformed_record','wrong_length'])
def test_failure_stops_new_launches_preserves_inflight_completion_and_never_retries(tmp_path,monkeypatch,failure):
    stage,launches,publications=fake_stage(tmp_path,monkeypatch,
        fail_first=failure=='nonzero_exit',broken_artifact=failure=='missing_artifact',
        malformed_record=failure=='malformed_record',wrong_length=failure=='wrong_length')
    ledger=driver.launch('runs/stage/seal.json','a'*40,poll_seconds=0)
    assert ledger['terminal'] and not ledger['completed']
    assert [j['state'] for j in ledger['jobs']]==['failed','completed','queued']
    assert len(launches)==2 and len(publications)==1
    if failure=='missing_artifact':assert ledger['jobs'][0]['completion_validation_error'].startswith('FileNotFoundError')
    if failure=='malformed_record':assert ledger['jobs'][0]['completion_validation_error'].startswith('ValueError')
    assert json.loads((stage/'execution.json').read_text())==ledger
    with pytest.raises(FileExistsError,match='never restart'):driver.launch('runs/stage/seal.json','a'*40,poll_seconds=0)
    assert len(launches)==2


def test_all_success_requires_every_unique_worker_to_complete_and_hash_check(tmp_path,monkeypatch):
    _,launches,_=fake_stage(tmp_path,monkeypatch)
    ledger=driver.launch('runs/stage/seal.json','a'*40,poll_seconds=0)
    assert ledger['completed'] and ledger['terminal']
    assert [j['state'] for j in ledger['jobs']]==['completed']*3
    assert len(launches)==3 and len({tuple(c) for c in launches})==3
    assert ledger['max_parallel_workers']==2


@pytest.mark.parametrize('audit_passed',[False,True])
def test_first_independent_audit_controls_parallel_admission_without_an_extra_physical_case(tmp_path,monkeypatch,audit_passed):
    stage,launches,_=fake_stage(tmp_path,monkeypatch,first_gate=True,audit_passed=audit_passed)
    ledger=driver.launch('runs/stage/seal.json','a'*40,poll_seconds=0)
    assert ledger['first_cell_audit_passed'] is audit_passed
    assert json.loads((stage/'first-cell-audit-v01.json').read_text())['passed'] is audit_passed
    assert ledger['first_cell_audit_sha256']==driver.digest(stage/'first-cell-audit-v01.json')
    assert len(launches)==(3 if audit_passed else 1)
    assert ledger['completed'] is audit_passed
    if not audit_passed:assert [j['state'] for j in ledger['jobs']]==['failed','queued','queued']
