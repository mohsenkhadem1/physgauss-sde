import json
from pathlib import Path
import pytest
from scripts.compositional_neural_mechanics.contact_transfer import collect_declared,sha
from scripts.compositional_neural_mechanics.contact_transfer_render import qualification_label


def test_timeout_preserves_completed_jobs_and_explicit_missing(tmp_path):
    folder=tmp_path/'states';folder.mkdir();key='mr_saturated-analytic_teacher-n6-q16';npz=folder/(key+'.npz');npz.write_bytes(b'testfixture')
    row=dict(key=key,family='mr_saturated',method='analytic_teacher',n=6,q=16,plan_sha256='fixture-plan',completed=True,response_sha256=sha(npz));(folder/(key+'.json')).write_text(json.dumps(row))
    rows,missing=collect_declared('fine',tmp_path,'fixture-plan')
    assert len(rows)==16 and len(missing)==15 and sum(x['completed']for x in rows)==1
    npz.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed'):collect_declared('fine',tmp_path,'fixture-plan')


def test_partial_render_never_claims_qualification():
    report=dict(complete=False,stage='fine',qualification={'mr_saturated_shared_laws':{'passed':True}})
    assert qualification_label(report,'mr_saturated','shared_laws')=='partial/unqualified'
    report['complete']=True
    assert qualification_label(report,'mr_saturated','field_difference')=='unqualified'


def test_gpu_launch_gate_rejects_unrelated_job_and_low_memory(monkeypatch):
    from scripts.compositional_neural_mechanics.contact_transfer import gpu_gate
    import subprocess
    state={'free':9000,'apps':''}
    def command(args,**kwargs):
        if args[0]=='systemctl':return 'ActiveState=inactive\nMainPID=0\nControlGroup=\n'
        if '--query-compute-apps=gpu_uuid,pid,process_name,used_memory'in args:return state['apps']
        return f"0, GPU-a, 11264, {state['free']}, 0\n1, GPU-b, 11264, 10000, 0\n"
    monkeypatch.setattr(subprocess,'check_output',command)
    assert gpu_gate()['census']['minimum_free_MiB']==8192
    state['free']=7000
    with pytest.raises(ValueError,match='headroom'):gpu_gate()
    state['free']=9000;state['apps']='GPU-a, 123, other-training, 256\n'
    with pytest.raises(ValueError,match='unrelatedjob'):gpu_gate()
