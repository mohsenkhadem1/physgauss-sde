"""Kernel-RSS/quick-exit stress and chained completed-arm reuse; no actualdata."""
import json,os,subprocess,sys,time
from pathlib import Path
import pytest
from experiments import run_campaign_c38_initial_inference_v03 as m
from experiments import run_campaign_c38_initial_inference_v02 as old
import test_run_campaign_c38_initial_inference as base_tests
import test_run_campaign_c38_initial_inference_v02 as repair_tests


def test_statm_measured_zero_vanished_and_malformed(monkeypatch):
 def read(text):monkeypatch.setattr(Path,'read_text',lambda *a,**k:text)
 read('123 11 0 0 0 0 0\n');assert m.current_rss(123)==11*os.sysconf('SC_PAGE_SIZE')
 read('0 0 0 0 0 0 0\n');assert m.current_rss(123)==0
 read('bad statm')
 with pytest.raises(RuntimeError,match='Malformed'):m.current_rss(123)
 def gone(*a,**k):raise FileNotFoundError('exited')
 monkeypatch.setattr(Path,'read_text',gone);assert m.current_rss(123) is None


def test_actual100_owned_quick_exits_and_live80MiB_rss(tmp_path):
 # Actual Linux proc sampling races; every child is owned/reaped, no namescan.
 for i in range(100):
  p=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
  try:
   for _ in range(8):
    value=m.current_rss(p.pid);assert value is None or value>=0
    if p.poll() is not None:break
   assert p.wait(timeout=3)==0
   value=m.current_rss(p.pid);assert value is None
  finally:m.cleanup_owned(p)
 ready=tmp_path/'ready';p=subprocess.Popen([sys.executable,'-c',"import sys,time;from pathlib import Path;x=bytearray(80*1024**2);Path(sys.argv[1]).write_text('ready');time.sleep(20)",str(ready)],start_new_session=True)
 try:
  deadline=time.monotonic()+3
  while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
  assert ready.exists() and m.current_rss(p.pid)>80*1024**2
 finally:m.cleanup_owned(p)
 assert p.poll() is not None and not Path('/proc',str(p.pid)).exists()


def test_chained_finished_arms_include_negative_and_all_hashes(tmp_path,monkeypatch):
 # Reuse the previous manufactured files; old validation supplies a synthetic
 # v02 closure, including a previously reused arm whose bytes live under v01.
 monkeypatch.setattr(repair_tests,'m',m)
 a,b,ids=repair_tests.reusable(tmp_path,monkeypatch)
 failed=json.loads((tmp_path/b['path']).read_text());failed['primary_failure']['message']='Truly live owned RSS measurement unavailable';(tmp_path/b['path']).write_text(json.dumps(failed));b=m.record(tmp_path,tmp_path/b['path'])
 before=dict(output='old',objects=[dict(object_id=t) for t in ids],model_inputs={'manufactured_public_only':True},continuation={'models_by_object':{}})
 # First collect no reuse nesting, then make one model a legitimate prior-ref.
 monkeypatch.setattr(old,'validate',lambda *a,**k:before)
 r=m.collect_reuse(tmp_path,a,b);assert r['reused_model_executions']==3
 v=m.reused_model(tmp_path,r['models_by_object'][ids[1]][m.MODELS[0]]);assert not v['completed'] and v['reused']
 descriptor=r['models_by_object'][ids[0]][m.MODELS[0]]
 path=tmp_path/descriptor['source_result']['path'];orig=json.loads(path.read_text())
 # New object-result metadata points to the old arm; prior closure fixes its
 # exact source record and payload. It cannot be silently changed/swapped.
 source=m.record(tmp_path,path);model=orig['models'][m.MODELS[0]]
 before['continuation']['models_by_object'][ids[0]]={m.MODELS[0]:dict(source_result=source,model=model)}
 orig['models'][m.MODELS[0]]=dict(model,reused=True,reused_from_result=source);path.write_text(json.dumps(orig))
 failed['result_lineage'][0]['result']=m.record(tmp_path,path);(tmp_path/b['path']).write_text(json.dumps(failed));b=m.record(tmp_path,tmp_path/b['path'])
 r=m.collect_reuse(tmp_path,a,b);assert r['reused_model_executions']==3
 (tmp_path/model['files']['numerics.json']['path']).write_bytes(b'tamper')
 with pytest.raises(ValueError,match='Changed input'):m.collect_reuse(tmp_path,a,b)


def test_exited_rss_sentinel_reaps_without_queue_abort(tmp_path,monkeypatch):
 monkeypatch.setattr(base_tests,'m',m);p,calls,cleaned=base_tests.fake_controller(tmp_path,monkeypatch);p['continuation']=dict(models_by_object={})
 original=m.subprocess.Popen
 class Race(original):
  def poll(self):return None
  def wait(self,timeout=None):return self.code
 monkeypatch.setattr(m.subprocess,'Popen',Race);monkeypatch.setattr(m,'current_rss',lambda pid:None)
 state=m.run('p.json','c'*64,root=tmp_path)
 assert state['completed'] and not state['technical_failure'] and len(calls)==64 and not cleaned
