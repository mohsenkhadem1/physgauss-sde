"""No actual study access: terminal barriers, lineage selection and tamper tests."""
import hashlib,json
from pathlib import Path
import pytest
from experiments import audit_fresh64_numerical_continuation_v01 as m


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value))


def test_root_predeclared_audit_pin_required_before_any_array(tmp_path,monkeypatch):
    monkeypatch.setattr(m.run,'LOCAL',tmp_path)
    monkeypatch.setattr(m.run,'release_gate',lambda _:({},{}))
    monkeypatch.setattr(m,'audit_predeclared_sources',lambda:{'auditor':'correct'})
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('Array read before terminal admission'))
    write(tmp_path/'release.json',dict(audit_source_SHA256={'auditor':'wrong'}))
    with pytest.raises(ValueError,match='prepinned'):m.terminal_metadata('p','e','s')


def test_incomplete_terminal_and_wrong_summary_hash_never_open_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(m.run,'LOCAL',tmp_path)
    monkeypatch.setattr(m.run,'release_gate',lambda _:({},{}))
    monkeypatch.setattr(m,'audit_predeclared_sources',lambda:{'a':'h'})
    monkeypatch.setattr(m.np,'load',lambda *a,**k:pytest.fail('Array read before all terminal seals'))
    write(tmp_path/'release.json',dict(audit_source_SHA256={'a':'h'}));write(tmp_path/'execution.json',dict(terminal=False));write(tmp_path/'summary-LOCAL.json',{})
    with pytest.raises(ValueError,match='terminal protocol'):m.terminal_metadata('p',digest(tmp_path/'execution.json'),digest(tmp_path/'summary-LOCAL.json'))
    with pytest.raises(ValueError,match='terminal/summary'):m.terminal_metadata('p',digest(tmp_path/'execution.json'),'wrong')


def test_raw_hash_or_path_tamper_cannot_fall_back_to_original(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    def record(path):return dict(path=str(Path(path).relative_to(tmp_path)),sha256=digest(path))
    monkeypatch.setattr(m,'record',record)
    path=tmp_path/'refined.npz';path.write_bytes(b'original synthetic bytes');binding=record(path);manifest={str(path):binding['sha256']}
    assert m.bound_archive(binding,manifest)==path
    path.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='terminal-bound'):m.bound_archive(binding,manifest)
    with pytest.raises(ValueError,match='Canonical'):m.bound_archive(dict(path='../original.npz',sha256='x'),manifest)
    with pytest.raises(ValueError,match='terminal-bound'):m.bound_archive(dict(path='missing-original.npz',sha256='x'),manifest)


def numerical(axis):
    return dict(qualified=False,diagnostics=dict(successive={a:[dict(passed=a!=axis)]for a in('inner','outer')}),strict_absolute_checks=dict(final_axis_witnesses={a:dict(passed=a!=axis)for a in('inner','outer')}))


@pytest.mark.parametrize('axis',['inner','outer'])
def test_refinement_choice_depends_only_on_failed_numerical_axis(axis):
    class NoScience(dict):
        def get(self,key,*args):
            if key in('PIT','covered','true_unit','target_RMS_m'):raise AssertionError('Scientific value accessed for plan selection')
            return super().get(key,*args)
    row=NoScience(numerical(axis));plan={a:[128,256,512]+([1024]if a==axis else[])for a in('inner','outer')};plan['moment_chunk']=1024
    entry=dict(decision='refine',failed_axes=[axis],plan=plan);m.census_selection(entry,row)
    bad=dict(entry,plan=dict(inner=[128,256,512,1024],outer=[128,256,512,1024],moment_chunk=1024))
    with pytest.raises(ValueError,match='only failed'):m.census_selection(bad,row)
    with pytest.raises(ValueError,match='qualified'):m.census_selection(dict(decision='reuse_qualified'),row)


def unresolved_memberships():
    rows=[];entries=[]
    for i in range(64):
        rows.append(dict(index=i,posteriors={k:{p:dict(qualified=False)for p in m.old.KEYS}for k in m.old.MODELS},posterior_lineage={k:{p:dict(accepted=None)for p in m.old.KEYS}for k in m.old.MODELS}))
        entries.extend(dict(id=f'{i:02d}/{k}/{p}')for k in m.old.MODELS for p in m.old.KEYS)
    return rows,entries


def test_summary_count_or_unblocked_flag_tamper_rejected_before_arrays():
    rows,entries=unresolved_memberships();s=dict(qualified_posteriors=0,unresolved=896,reused_qualified=0,newly_qualified=0,comparison_blocked=True,completed=True);e=dict(completed=True,workers=[dict(reaped=True,exit_code=0)])
    assert m.summary_membership_checks(s,e,rows,entries,1)['comparison_blocked']
    for key,value in [('qualified_posteriors',1),('comparison_blocked',False),('newly_qualified',1),('completed','true')]:
        with pytest.raises(ValueError):m.summary_membership_checks(dict(s,**{key:value}),e,rows,entries,1)
    with pytest.raises(ValueError):m.summary_membership_checks(s,e,rows[:-1],entries,1)
