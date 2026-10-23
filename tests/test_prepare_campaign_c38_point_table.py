"""Manufactured80-field metadata/point arrays only; no campaign captures read."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments import prepare_campaign_c38_point_table as m


def record(root,p):return dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def fixtures(root):
    rows=[];index=[];reuse=[]
    for i,c in enumerate(m.catalog()):
        meshes=[]
        for nx in (32,40):
            p=root/f'field-{i}-{nx}.npz';x=75000/c['young_Pa'];Q=m.POINTS.copy();Q[:,0]+=.001*(1+c['Fz_N']/10)*x+.00005*x*x
            np.savez(p,material_probes_m=m.POINTS,probe_positions_m=Q,positions_m=np.array(['FORBIDDEN_RAW_FULL_FIELD']))
            meshes.append(dict(cells=[nx,nx//2,nx//2],capture=record(root,p),completed=True,physics_passed=True))
        s=root/f'case-{i}.json';cache=root/f'cache-{i}.json';data=dict(case=c,completed=True,passed=True,physics_passed=True,meshes=meshes);s.write_text(json.dumps(data));cache.write_bytes(s.read_bytes())
        sr,cr=record(root,s),record(root,cache)
        rows.append(dict(case=c,source_result=sr,external_cache=cr,completed=True,passed=True,physics_passed=True,meshes=meshes))
        if c['reused_original14_case']:reuse.append(dict(case_id=c['id'],source_result=sr,external_cached_result=cr))
        else:index.append(dict(path=sr['path'],sha256=sr['sha256'],cache=cr['path'],completed=True,passed=True))
    summary=dict(completed=True,all40_declared_q63_static_gates_passed=True,rows=rows)
    return summary,dict(records=index),dict(reused_case_records=reuse)


def test_exact80_global_hash_barrier_and_only_two_point_members_extracted(tmp_path,monkeypatch):
    summary,index,pre=fixtures(tmp_path);fields=m.plan_completed_rows(summary,root=tmp_path);m.verify_external_index(fields,index,pre)
    calls=[];original=np.lib.npyio.NpzFile.__getitem__
    def tracked(self,key):
        calls.append(key);assert key in ('material_probes_m','probe_positions_m');return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',tracked)
    rows=m.load_point_rows(fields,root=tmp_path)
    assert len(rows)==80 and calls==['material_probes_m','probe_positions_m']*80
    assert [r['source_index'] for r in rows]==list(range(80))
    result=m.validate_finite_table(rows,tolerance_m=m.TOLERANCE_M)
    assert result['finite_checks_passed'] and len(result['checks'])==16
    assert result['maximum_withheld_point_error_m']<1e-15


def test_last_capture_tamper_rejected_before_any_array_extraction(tmp_path,monkeypatch):
    summary,_,_=fixtures(tmp_path);fields=m.plan_completed_rows(summary,root=tmp_path)
    (tmp_path/fields[-1]['capture']['path']).write_bytes(b'tamper')
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Arrays read before fullhashbarrier')))
    with pytest.raises(ValueError,match='Changed'):m.load_point_rows(fields,root=tmp_path)


def test_roles_remain_separate_failures_retained_without_refit_or_promotion(tmp_path):
    summary,_,_=fixtures(tmp_path);fields=m.plan_completed_rows(summary,root=tmp_path);rows=m.load_point_rows(fields,root=tmp_path)
    withheld=[r for r in rows if r['case']['material_use']=='WITHHELD_INTERPOLATION_VALIDATION'];withheld[0]['probe_positions_m'][3,2]+=1e-5
    rows[0]['finite_spatial_passed']=False
    result=m.validate_finite_table(rows,tolerance_m=1e-6)
    assert not result['all_withheld_interpolation_checks_passed'] and not result['all_direct_finite_physical_and_spatial_checks_passed'] and not result['finite_checks_passed']
    assert len(result['checks'])==16 and any(not r['passed'] for r in result['checks'])
    archives=m.split_archives(rows);assert len(archives['TABLE_NODE']['source_index'])==48 and len(archives['WITHHELD_INTERPOLATION_VALIDATION']['source_index'])==32
    assert not set(archives['TABLE_NODE']['source_index'])&set(archives['WITHHELD_INTERPOLATION_VALIDATION']['source_index'])
    assert set(archives['TABLE_NODE']['material_use'])=={'TABLE_NODE'}
    assert set(archives['WITHHELD_INTERPOLATION_VALIDATION']['material_use'])=={'WITHHELD_INTERPOLATION_VALIDATION'}


def test_missing_duplicate_promoted_role_or_bad_external_cache_fails(tmp_path):
    summary,index,pre=fixtures(tmp_path)
    bad=copy.deepcopy(summary);bad['rows'].pop()
    with pytest.raises(ValueError,match='exact40'):m.plan_completed_rows(bad,root=tmp_path)
    bad=copy.deepcopy(summary);bad['rows'][1]['case']=bad['rows'][0]['case']
    with pytest.raises(ValueError,match='exact40'):m.plan_completed_rows(bad,root=tmp_path)
    bad=copy.deepcopy(summary);next(r for r in bad['rows'] if r['case']['material_use']=='WITHHELD_INTERPOLATION_VALIDATION')['case']['material_use']='TABLE_NODE'
    with pytest.raises(ValueError,match='fixed source-index'):m.plan_completed_rows(bad,root=tmp_path)
    fields=m.plan_completed_rows(summary,root=tmp_path);index['records'].pop()
    with pytest.raises(ValueError,match='terminal-cache'):m.verify_external_index(fields,index,pre)


def test_recipe_and_future_prepare_use_tempROOT_exclusive_no_arrays(tmp_path,monkeypatch):
    summary,index,pre=fixtures(tmp_path)
    for name,data in ((m.SUMMARY,summary),(m.CACHE,index),(m.PREREQUISITE,pre)):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
    for name in m.source_names():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((m.ROOT/name).read_bytes())
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Arrays read during preparation')))
    original=Path.open;writes=[]
    def guard(self,mode='r',*a,**kw):
        if any(c in mode for c in 'wax+'):
            assert self.resolve().is_relative_to(tmp_path.resolve());writes.append(self)
        return original(self,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    m.write_recipe('opdev/recipe.json',root=tmp_path)
    digest=m.prepare_execution('opdev/protocol.json','future-output',root=tmp_path)
    p=m.validate_execution('opdev/protocol.json',digest,root=tmp_path)
    assert len(p['fields'])==80 and p['tolerance_m']==1e-6 and len(writes)==2 and not (tmp_path/'future-output').exists()
    with pytest.raises(ValueError,match='Fresh exclusive'):m.prepare_execution('opdev/protocol.json','future-output',root=tmp_path)
