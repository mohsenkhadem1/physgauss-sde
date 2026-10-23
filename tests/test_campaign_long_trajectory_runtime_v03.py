import json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import pytest
from experiments.campaign_long_trajectory_runtime_v03 import JOBS,GPUS,checkpoint,load_checkpoint,validate_lease,verify_pins,supervise,atomic_json,watchdog


def test_checkpoints_preserve_two_committed_states_ignore_uncommitted_and_reject_corruption(tmp_path):
    u=np.zeros((3,3));v=np.ones_like(u)
    for k in range(3):checkpoint(tmp_path,u+k,v,k,k/32,'pin','job',accepted_ledger={'passed':True})
    assert len(list(tmp_path.glob('accepted-*')))==2
    (tmp_path/'accepted-99999').mkdir() # interrupted uncommitted generation never read
    a,b,m=load_checkpoint(tmp_path,'pin','job');np.testing.assert_array_equal(a,u+2);assert m['accepted_ledger']['passed']
    with pytest.raises(ValueError):load_checkpoint(tmp_path,'wrong','job')
    with pytest.raises(ValueError):checkpoint(tmp_path,u,v,4,.2,'pin','job')
    current=tmp_path/'accepted-00002'/'state.npz';current.write_bytes(current.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):load_checkpoint(tmp_path,'pin','job')


def test_wrong_GPU3_CPU_or_input_pin_is_rejected_before_worker(tmp_path):
    validate_lease(JOBS[0],GPUS[0],[0,1])
    for gpu,cores in [(GPUS[1],[0,1]),('GPU-9dbe9939-379d-9560-2028-462a2f8b3ac1',[0,1]),(GPUS[0],[2,3])]:
        with pytest.raises(ValueError):validate_lease(JOBS[0],gpu,cores)
    path=tmp_path/'source.py';path.write_text('immutable')
    with pytest.raises(ValueError):verify_pins({str(path):'wrong'})


def test_fixed_pipeline_reaps_pass_before_slot_reuse_and_keeps_all_four(tmp_path):
    spawned=[];children={}
    def spawn(j):
        if j['id']==JOBS[3]['id']:assert children[JOBS[0]['id']].returncode==0
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.15)'],start_new_session=True)
        spawned.append(j['id']);children[j['id']]=child;return child
    assert supervise(list(JOBS),spawn,lambda j:True,global_cap_s=3,per_job_cap_s=2,receipt_path=tmp_path/'execution.json',poll_s=.01)
    assert spawned==[j['id'] for j in JOBS]
    receipt=json.loads((tmp_path/'execution.json').read_text());assert receipt['terminal'] and len(receipt['completed'])==4
    assert all(c.returncode==0 and not Path('/proc/'+str(c.pid)).exists() for c in children.values())


def test_timeout_cancels_queued_and_reaps_TERM_ignoring_owned_group(tmp_path):
    children=[]
    def spawn(j):
        child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],start_new_session=True)
        children.append(child);return child
    assert not supervise(list(JOBS),spawn,lambda j:True,global_cap_s=.1,per_job_cap_s=10,receipt_path=tmp_path/'timeout.json',poll_s=.01)
    r=json.loads((tmp_path/'timeout.json').read_text());assert r['terminal'] and r['failed'] and len(children)==3
    assert all(c.returncode is not None and not Path('/proc/'+str(c.pid)).exists() for c in children)


def test_numerical_failure_cancels_queued_and_reaps_active_peers(tmp_path):
    children=[]
    def spawn(j):
        code='import time;time.sleep(.05)' if j['slot']==0 else 'import time;time.sleep(20)'
        child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True);children.append(child);return child
    assert not supervise(list(JOBS),spawn,lambda j:False,global_cap_s=3,per_job_cap_s=2,receipt_path=tmp_path/'failure.json',poll_s=.01)
    assert len(children)==3 and all(c.returncode is not None for c in children)


def test_watchdog_unblocks_inherited_TERM_and_enforces_native_step_cap(tmp_path):
    # Separate real child ignores TERM, simulating a native-blocked worker.
    child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],start_new_session=True)
    import multiprocessing
    deadline=tmp_path/'deadline.json';atomic_json(deadline,{'deadline_monotonic_s':time.monotonic()+.15})
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
    try:
        guard=multiprocessing.get_context('fork').Process(target=watchdog,args=(child.pid,time.monotonic()+3,deadline,10**12,tmp_path/'guard.json',.01,.05));guard.start()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
    try:
        child.wait(timeout=3);guard.join(timeout=2)
        assert child.returncode!=0 and not guard.is_alive()
        assert json.loads((tmp_path/'guard.json').read_text())['reason']=='STEP_WALL_LIMIT'
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if guard.is_alive():guard.terminate();guard.join()


def test_cleanup_drains_owned_descendant_and_preserves_unrelated_group(tmp_path):
    children=[];unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    def spawn(j):
        pidfile=tmp_path/(j['id']+'.pid')
        program='''import subprocess,sys,signal,time,pathlib
c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
pathlib.Path(sys.argv[1]).write_text(str(c.pid))
def stop(*args):
 c.wait(timeout=1)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
time.sleep(20)
'''
        child=subprocess.Popen([sys.executable,'-c',program,str(pidfile)],start_new_session=True);children.append(child);return child
    try:
        assert not supervise(list(JOBS),spawn,lambda j:True,global_cap_s=.4,per_job_cap_s=10,receipt_path=tmp_path/'groups.json',poll_s=.01)
        assert unrelated.poll() is None and len(children)==3
        pids=[int(p.read_text()) for p in tmp_path.glob('*.pid')]
        assert len(pids)==3 and all(not Path('/proc/'+str(pid)).exists() for pid in pids)
    finally:
        unrelated.terminate();unrelated.wait()
