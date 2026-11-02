"""Supervisor fixtures only: no simulations, data draws, or GPU jobs."""
import json,sys,time
import pytest
from scripts.compositional_dynamics_v17 import cohort_pipeline as m

@pytest.mark.parametrize('command,timeout,success', [([sys.executable,'-c','print("done")'],2,True),([sys.executable,'-c','raise SystemExit(3)'],2,False),([sys.executable,'-c','import time;time.sleep(5)'],.1,False)])
def test_child_durable_exit_and_timeout(tmp_path,monkeypatch,command,timeout,success):
 monkeypatch.setattr(m,'checked',lambda x:None)
 request=tmp_path/'request.json';status=tmp_path/'status.json'
 m.write(request,dict(bindings={},stage='fixture',status_path=str(status),log_path=str(tmp_path/'log'),command=command,timeout_seconds=timeout,deadline_epoch=time.time()+3))
 m.child(request);result=json.loads(status.read_text())
 assert result['state']=='terminal'and result['completed']is success
 assert result['request_sha256']==m.sha(request)

def test_resume_terminal_no_reexecution(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'PIPE',tmp_path);directory=tmp_path/'stages'/'one';deadline=time.time()+10
 m.write(directory/'request.json',dict(command=['never'],bindings={},deadline_epoch=deadline))
 m.write(directory/'status.json',dict(state='terminal',completed=False,request_sha256=m.sha(directory/'request.json')))
 assert m.run_stage('one',['never'],1,{},deadline)['completed']is False
 with pytest.raises(ValueError,match='differs'):m.run_stage('one',['different'],1,{},deadline)

def test_native_failure_all_identity_and_preservation(tmp_path):
 bindings=dict(protocol_sha='p',freeze_sha='f')
 row=m.native_commit('world-0','forecast-40mm',bindings,tmp_path,'Calibration unavailable')
 assert row['completed']is False and row['method_freeze_sha256']=='f'
 assert m.native_commit('world-0','forecast-40mm',bindings,tmp_path)==row
 with pytest.raises(ValueError,match='corrupted'):m.native_commit('world-1','forecast-40mm',bindings,tmp_path)

def test_concurrent_stage_callers_serialize_creation(tmp_path,monkeypatch):
 import concurrent.futures,threading
 monkeypatch.setattr(m,'PIPE',tmp_path)
 state={'active':0,'peak':0,'created':0};counter=threading.Lock()
 def private(stage,*args):
  with counter:state['active']+=1;state['peak']=max(state['peak'],state['active'])
  marker=tmp_path/'created'
  if not marker.exists():marker.write_text('one');state['created']+=1
  time.sleep(.02)
  with counter:state['active']-=1
  return {'completed':True}
 monkeypatch.setattr(m,'_run_stage',private)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
  results=list(pool.map(lambda _:m.run_stage('export-world-0',['dummy'],1,{},time.time()+5),range(2)))
 assert state['peak']==1 and state['created']==1 and all(r['completed']for r in results)

def test_final_census_keeps_fixed16_and_preserves_corruption(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path)
 p={'generator_settings':{'world_ids':[f'world-{i}'for i in range(4)]},'native_worlds':[dict(name=f'world-{i}',output_directory=f'native/world-{i}')for i in range(4)]}
 recipe={'forecast_root':'shared'};bindings=dict(protocol_sha='p',freeze_sha='f')
 corrupt=tmp_path/'shared/world-0/forecast-40mm/prediction-commit.json';m.write(corrupt,{'completed':True,'protocol_sha256':'wrong'});digest=m.sha(corrupt)
 census=m.final_census(p,recipe,bindings,{'lanes':{}})
 assert census['present']==16 and not census['all_identities_valid'] and len(census['errors'])==1
 assert m.sha(corrupt)==digest
 assert sum(r['completed']is False for r in census['records'])==15
