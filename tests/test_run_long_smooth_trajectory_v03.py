import json,os,signal,subprocess,sys
from pathlib import Path
import pytest
from experiments import run_long_smooth_trajectory_v03 as driver


def test_prepare_uses_temp_namespace_exclusive_and_pins_sources_without_array_access(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'OP',tmp_path/'operator')
    monkeypatch.setattr(driver.np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Prepare touched field arrays')))
    digest=driver.prepare();p=driver.validate(digest)
    assert p['resources']['global_wall_cap_s']==13500 and p['resources']['per_trajectory_wall_cap_s']==12600
    assert len(p['jobs'])==4 and len(p['GPUs'])==3 and not p['admission']
    with pytest.raises(FileExistsError):driver.prepare()
    with pytest.raises(ValueError):driver.validate('wrong')
    p['source_pins'][str(tmp_path/'missing')]='bad';driver.atomic_json(driver.OP/'protocol.json',p)
    with pytest.raises(ValueError):driver.validate(driver.sha(driver.OP/'protocol.json'))


def test_signal_restore_and_alarm_after_real_exec_with_inherited_block_mask(tmp_path):
    program='''import signal,time
from experiments.run_long_smooth_trajectory_v03 import install_worker_limits
install_worker_limits(2,.05)
try:time.sleep(5)
except TimeoutError:print("RESTORED_ALARM",flush=True)
else:raise RuntimeError("alarm masked")
'''
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
    try:child=subprocess.Popen([sys.executable,'-c',program],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
    stdout,stderr=child.communicate(timeout=4)
    assert child.returncode==0 and 'RESTORED_ALARM' in stdout,stderr


def test_full_four_bundle_aggregate_blocks_missing_last_state_and_velocity_failure(tmp_path,monkeypatch):
    import numpy as np
    from experiments.campaign_long_trajectory_runtime_v03 import checkpoint,record
    monkeypatch.setattr(driver,'OUT',tmp_path/'output');monkeypatch.setattr(driver,'OP',tmp_path/'operator')
    driver.OP.mkdir();driver.OUT.mkdir();p={'clocks_s':(np.arange(9)/8).tolist()};digest='frozen-synthetic'
    def bundle(job,velocity_bad=False):
        folder=driver.OUT/job['id'];folder.mkdir(exist_ok=True);clocks=[];n=job['steps']
        u=np.zeros((3,3))+1/n**2*1e-3;v=np.zeros_like(u)+1/n**2*2e-3
        if job['cells'][0]==32:u+=1e-7;v+=2e-7
        if velocity_bad:v+=.1
        for k,t in enumerate(p['clocks_s']):
            path=folder/f'clock{k}.npz';np.savez(path,q=u,u=u,v=v,probe_q=u,probe_v=v);clocks.append({'time_s':t,'capture':record(path)})
        ledger=folder/'ledger.jsonl';ledger.write_text(''.join(json.dumps({'step':k,'time_s':k/n,'passed':True})+'\n' for k in range(1,n+1)))
        geometry=folder/'geometry.json';geometry.write_text('{}')
        if not (folder/'checkpoints').exists():
            for k in range(n+1):checkpoint(folder/'checkpoints',u,v,k,k/n,digest,job['id'])
        result=dict(completed=True,passed=True,job=job,accepted_steps=n,protocol_sha256=digest,clocks=clocks,hmax_m=.02 if job['cells'][0]==32 else .016,accepted_ledger=record(ledger),geometry=record(geometry),final_checkpoint_commit=record(folder/'checkpoints'/'COMMIT.json'))
        driver.atomic_json(folder/'result.json',result)
    for j in driver.JOBS[:3]:bundle(j)
    with pytest.raises(FileNotFoundError):driver.aggregate(p,digest)
    assert not (driver.OP/'convergence-summary.json').exists()
    bundle(driver.JOBS[3]);assert driver.aggregate(p,digest)['conditional_convergence_passed']
    bundle(driver.JOBS[3],velocity_bad=True)
    report=driver.aggregate(p,digest);assert not report['conditional_convergence_passed'] and not report['admission']
    path=driver.OUT/driver.JOBS[3]['id']/'ledger.jsonl';path.write_text(path.read_text()+'corrupt')
    with pytest.raises(ValueError):driver.aggregate(p,digest)
