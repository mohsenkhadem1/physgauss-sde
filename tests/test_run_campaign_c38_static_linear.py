"""CPU metadata/seal controls only; no GPU/linear/physical solves."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_campaign_c38_static_linear as m


def prepare_fixture(tmp,monkeypatch):
    original=m.ROOT
    for name in m.source_closure()+[m.BASE_PROTOCOL]:
        p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((original/name).read_bytes())
    monkeypatch.setattr(m,'ROOT',tmp)
    protocol=tmp/'opdev'/'protocol.json';output=tmp/'output'
    return protocol,output


def test_prepare_isolated_exclusive_full_closure_and_exact_physical_jobs(tmp_path,monkeypatch):
    protocol,out=prepare_fixture(tmp_path,monkeypatch)
    original_open=Path.open;writes=[]
    def guarded(self,mode='r',*a,**kw):
        if any(flag in mode for flag in 'wax+'):
            assert self.resolve().is_relative_to(tmp_path.resolve());writes.append(self)
        return original_open(self,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    digest=m.prepare(protocol,out);p=m.validate(protocol,digest)
    assert writes==[protocol] and not out.exists()
    assert len(p['jobs'])==4 and len(p['all_cases'])==14
    assert p['nonlinear_target_access'] is False and p['scoring_authorized'] is False
    assert p['reference_young_Pa']==100000 and p['resources']['global_wall_cap_s']==600
    assert set(p['source_sha256'])==set(m.source_closure())
    with pytest.raises(ValueError,match='Fresh'):m.prepare(protocol,out)


def test_altered_design_or_source_rejected(tmp_path,monkeypatch):
    protocol,out=prepare_fixture(tmp_path,monkeypatch);m.prepare(protocol,out)
    p=json.loads(protocol.read_text());p['nonlinear_target_access']=True
    protocol.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='barrier'):m.validate(protocol,m.sha(protocol))
    p['nonlinear_target_access']=False;protocol.write_text(json.dumps(p))
    (tmp_path/m.LINEAR_SOURCE).write_text('# corrupt baseline\n')
    with pytest.raises(ValueError,match='source closure'):m.validate(protocol,m.sha(protocol))


def prediction_fixture(tmp):
    results=[];cases=m.base.all_cases()
    for job in m.jobs():
        rows=[]
        for case in cases:
            if case['support']!=job['support']:continue
            path=tmp/(job['id']+'__'+case['id']+'.npz')
            probes=m.QUERY_POINTS+case['Fz_N']*1e-4
            if job['cells'][0]==40:probes=probes+1e-7
            np.savez_compressed(path,positions_m=np.zeros((2,3)),probe_positions_m=probes,material_probes_m=m.QUERY_POINTS)
            rows.append(dict(case=case,cells=job['cells'],capture={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},hmax_m=1/job['cells'][0],prediction_J_passed=True))
        results.append(dict(job=job,completed=True,passed=True,predictions=rows))
    return dict(jobs=m.jobs(),all_cases=cases),results


def test_all28_seal_exact_case_joins_spatial_indicator_and_J_failure_retained(tmp_path,monkeypatch):
    p,results=prediction_fixture(tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path)
    results[0]['predictions'][0]['prediction_J_passed']=False
    seal=m.seal_predictions(p,results)
    assert seal['completed'] and len(seal['rows'])==14
    assert seal['all28_predictions_sealed_before_nonlinear_target_access']
    assert not seal['nonlinear_target_access'] and not seal['all_prediction_J_passed']
    assert seal['all_linear_spatial_indicators_passed']
    assert abs(seal['rows'][0]['actual_h_factor']-4)<1e-14
    assert abs(seal['rows'][0]['conditional_linear_spatial_indicator_m']-4*np.sqrt(3)*1e-7)<1e-14
    bad=copy.deepcopy(results);bad[0]['predictions'].pop()
    with pytest.raises(ValueError,match='All4'):m.seal_predictions(p,bad)


def test_prediction_tamper_and_duplicate_case_resolution_fail_closed(tmp_path,monkeypatch):
    p,results=prediction_fixture(tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path)
    bad=copy.deepcopy(results);bad[0]['predictions'][0]['case']=bad[0]['predictions'][1]['case']
    with pytest.raises(ValueError,match='Exact case'):m.seal_predictions(p,bad)
    path=tmp_path/results[0]['predictions'][0]['capture']['path'];path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='capture changed'):m.seal_predictions(p,results)


def test_inverted_linear_prediction_is_measured_not_elastically_rejected():
    import torch
    class GeometricOperator:
        elements=torch.tensor([[0]])
        def chunks(self,checking=False):return [(None,None)]
        def deformation(self,local,grad):return torch.diag(torch.tensor([-1.,1.,1.],dtype=torch.float64))[None,None]
    calls=[]
    assert m.sampled_minimum_J(GeometricOperator(),torch.zeros((1,3),dtype=torch.float64),resource_guard=lambda:calls.append(1))==-1
    assert len(calls)==2
