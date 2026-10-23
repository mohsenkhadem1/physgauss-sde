"""Exit-race and hash-bound continuation controls; manufactured files only."""
import json
from pathlib import Path
import pytest
from experiments import run_campaign_c38_initial_inference_v02 as m
from experiments import run_campaign_c38_initial_inference as old
import test_run_campaign_c38_initial_inference as prior_tests


def test_proc_exit_states_vanished_and_truly_live_missing_rss(monkeypatch):
    original=Path.read_text
    def status(text):
        def read(self,*a,**k):
            if self.name=='status' and '/proc/' in str(self):return text
            return original(self,*a,**k)
        monkeypatch.setattr(Path,'read_text',read)
    status('State:\tZ (zombie)\n');assert m.current_rss(987654) is None
    status('State:\tX (dead)\n');assert m.current_rss(987654) is None
    status('State:\tR (running)\nVmRSS:\t123 kB\n');assert m.current_rss(987654)==123*1024
    status('State:\tS (sleeping)\n')
    with pytest.raises(RuntimeError,match='Truly live'):m.current_rss(987654)
    def vanished(self,*a,**k):raise FileNotFoundError('Exited between poll/status')
    monkeypatch.setattr(Path,'read_text',vanished);assert m.current_rss(987654) is None


def reusable(root,monkeypatch):
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return m.record(root,p)
    predecessor=put('old-protocol.json',{});ids=[format(i+1,'032x') for i in range(3)];lineage=[];rows=[]
    for i,token in enumerate(ids):
        row=dict(object_id=token,pid=100000+i);rows.append(row)
        if i==2:lineage.append(dict(object_id=token,complete_result_available=False));continue
        models={}
        for name in (m.MODELS if i==0 else m.MODELS[:1]):
            files={}
            for f in ('posterior-arrays.npz','posterior-structure.json','numerics.json'):
                files[f]=put('old/objects/'+token+'/'+name+'/'+f,{'manufactured':True})
            # A finished numerical negative MUST also be reused, never filtered.
            models[name]=dict(execution_completed=True,completed=i!=1,numerical_controls_passed=i!=1,files=files)
        r=put('old/objects/'+token+'/result.json',dict(completed=i==0,pid=row['pid'],object_id=token,protocol_sha256=predecessor['sha256'],models=models,private_truth_access=False))
        lineage.append(dict(object_id=token,result=r))
    execution=put('old/execution.json',dict(protocol=predecessor,completed=False,all_owned_workers_reaped=True,records=rows,queued_object_ids=['unlaunched']))
    failed=put('failed.json',dict(terminal=True,completed=False,protocol=predecessor,primary_failure={'message':'Owned RSS measurement unavailable'},controller_and_all12_worker_PIDs_absent=True,execution=execution,result_lineage=lineage))
    before=dict(output='old',objects=[dict(object_id=t) for t in ids],model_inputs={'manufactured_public_only':True})
    monkeypatch.setattr(old,'validate',lambda *a,**k:before)
    return predecessor,failed,ids


def test_every_completed_arm_including_numeric_negative_reused_and_no_arrays(tmp_path,monkeypatch):
    a,b,ids=reusable(tmp_path,monkeypatch)
    monkeypatch.setattr(m.np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Arrays in reuse metadata')))
    r=m.collect_reuse(tmp_path,a,b)
    assert r['fully_reused_objects']==1 and r['reused_model_executions']==3 and r['new_model_executions_required']==125
    assert r['interrupted_objects_requiring_continuation']==2 and r['no_observations_or_identities_regenerated']
    v=m.reused_model(tmp_path,r['models_by_object'][ids[1]][m.MODELS[0]])
    assert v['reused'] and not v['completed'] and not v['numerical_controls_passed']
    assert ids[2] not in r['models_by_object']
    file=v['files']['numerics.json'];(tmp_path/file['path']).write_text('changed')
    with pytest.raises(ValueError,match='Changed input'):m.collect_reuse(tmp_path,a,b)


def test_reused_result_identity_cannot_be_swapped(tmp_path,monkeypatch):
    a,b,ids=reusable(tmp_path,monkeypatch);r=m.collect_reuse(tmp_path,a,b)
    descriptor=r['models_by_object'][ids[0]][m.MODELS[0]]
    descriptor['model_name']=m.MODELS[1]
    # Different source refs are part of model, even if numeric flags equal.
    with pytest.raises(ValueError,match='Exact reused'):m.reused_model(tmp_path,descriptor)


def test_exit_between_poll_and_status_is_reaped_instead_of_aborting_queue(tmp_path,monkeypatch):
    monkeypatch.setattr(prior_tests,'m',m)
    p,calls,cleaned=prior_tests.fake_controller(tmp_path,monkeypatch)
    p['continuation']=dict(models_by_object={})
    # Model a real zombie race: poll saw live, RSS saw Z, wait reaps code0.
    original=m.subprocess.Popen
    class Race(original):
        def poll(self):return None
        def wait(self,timeout=None):return self.code
    monkeypatch.setattr(m.subprocess,'Popen',Race)
    monkeypatch.setattr(m,'current_rss',lambda pid:None)
    state=m.run('p.json','c'*64,root=tmp_path)
    assert state['completed'] and len(calls)==64 and not state['technical_failure'] and not cleaned
    assert all(r['reaped'] and r['exit_code']==0 for r in state['records'])


def test_all_complete_objects_reused_without_process_and_partial_arm_metadata_retained(tmp_path,monkeypatch):
    monkeypatch.setattr(prior_tests,'m',m);p,calls,cleaned=prior_tests.fake_controller(tmp_path,monkeypatch)
    token=p['objects'][0]['object_id'];saved={}
    for name in m.MODELS:
        saved[name]=dict(model=dict(execution_completed=True,completed=True,numerical_controls_passed=True,files={}))
    p['continuation']=dict(models_by_object={token:saved})
    def reuse(root,desc):
        # Manufactured retained outputs, representing already hash-validated refs.
        name=next(n for n in m.MODELS if saved[n] is desc);b=tmp_path/'reused'/name;b.mkdir(parents=True,exist_ok=True);files={}
        for f in ('posterior-arrays.npz','posterior-structure.json','numerics.json'):
            (b/f).write_bytes(b'preserved synthetic arm');files[f]=m.record(tmp_path,b/f)
        return dict(desc['model'],files=files,reused=True)
    monkeypatch.setattr(m,'reused_model',reuse)
    state=m.run('p.json','c'*64,root=tmp_path)
    assert state['completed'] and len(calls)==63
    reused=[r for r in state['records'] if r.get('kind')=='REUSED_COMPLETE_OBJECT_NO_PROCESS']
    assert len(reused)==1 and reused[0]['pid'] is None
