"""Independent no-physics proof of both replay wall-clock enforcement paths."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest
from scripts.coupled_contact_v13 import replay_surface_job as r


@pytest.mark.parametrize("child_alarm_wins",[False,True])
def test_supervisor_timeout_kills_dedicated_dummy_process_group(tmp_path,monkeypatch,child_alarm_wins):
    from scripts.coupled_contact_v13 import surface_pipeline
    report=tmp_path/'report';report.mkdir()
    (report/'protocol.json').write_text('{}');(report/'method-freeze.json').write_text('{}')
    monkeypatch.setattr(r,'REPORT',report)
    monkeypatch.setattr(r,'verify',lambda:({'physics_qualification_sha256':'test','source_sha256':{}},{}))
    monkeypatch.setattr(r,'select_job',lambda selector,p:({'path':selector},{}))
    monkeypatch.setattr(r,'environment',lambda:{})
    monkeypatch.setattr(surface_pipeline,'simulation_settings',lambda j,p:{'fixture':True})
    original=subprocess.Popen;started=[];marker=tmp_path/'should-not-exist'
    # A grandchild would create a marker if killing only the parent were used.
    code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1]]);time.sleep(10)"
    grandchild=f"import time;from pathlib import Path;time.sleep(.7);Path({str(marker)!r}).write_text('alive')"
    def dummy(command,**kwargs):
        dummycode=code
        if child_alarm_wins:
            dummycode=f"import os,signal,time;from pathlib import Path;time.sleep(.05);Path({str(tmp_path/'fresh'/'replay-timeout.json')!r}).write_text('{{}}');os.killpg(os.getpgrp(),signal.SIGKILL)"
        process=original([sys.executable,'-c',dummycode,grandchild],**kwargs);started.append(process);return process
    monkeypatch.setattr(r.subprocess,'Popen',dummy)
    out=tmp_path/'fresh';monkeypatch.setattr(sys,'argv',['replay','--job','library/fem/0-0','--output',str(out),'--max-seconds','.25'])
    begin=time.monotonic()
    try:
        assert r.main()==124
        assert time.monotonic()-begin<2
        result=json.loads((out/'replay-execution.json').read_text())
        assert result['timed_out'] and result['returncode']==-signal.SIGKILL
        time.sleep(.8)
        assert not marker.exists()
    finally:
        for process in started:
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()


def test_worker_alarm_kills_itself_without_supervisor_timeout(tmp_path):
    (tmp_path/'replay-request.json').write_text(json.dumps({'deadline_unix':time.time()+.7}))
    code="from scripts.coupled_contact_v13 import replay_surface_job as r;import time,sys;r.verify=lambda:time.sleep(10);r.worker(sys.argv[1])"
    process=subprocess.Popen([sys.executable,'-B','-c',code,str(tmp_path)],cwd=r.ROOT,start_new_session=True)
    begin=time.monotonic()
    try:
        assert process.wait(timeout=3)==-signal.SIGKILL
        assert time.monotonic()-begin<2
        marker=json.loads((tmp_path/'replay-timeout.json').read_text())
        assert marker['reason']=='Independent child wall-clock deadline'
        assert marker['observed_unix']>=marker['deadline_unix']
    finally:
        if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
