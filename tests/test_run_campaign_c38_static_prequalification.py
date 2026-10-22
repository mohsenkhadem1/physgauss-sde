import json
from pathlib import Path
import pytest
from experiments import run_campaign_c38_static_prequalification as r


def test_exact14_physical_cases_and_local_operator_source_closure():
    rows=r.cases();assert len(rows)==14 and len({v['id'] for v in rows})==14
    assert sum(v['kind']=='domain_corner' for v in rows)==12
    assert sum(v['kind']=='interior_gravity' and v['Fz_N']==0 for v in rows)==2
    assert min(v['young_Pa'] for v in rows)==75000 and max(v['young_Pa'] for v in rows)==150000
    names=r.source_closure()
    for name in ['experiments/campaign_p2_checkpointed.py','experiments/campaign_p2_keast_quadrature.py','experiments/campaign_p2_exact_quadrature.py','experiments/campaign_p2_supported_preconditioner.py','experiments/campaign_p2_jacobian_certificate.py','experiments/campaign_c38_static_family.py']:assert name in names
    assert len(names)==len(set(names)) and all((r.ROOT/n).is_file() for n in names)


def test_prospective_metadata_rejects_old_material_or_unknown_worker_before_physics(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'source_closure',lambda:['source.py']);(tmp_path/'source.py').write_text('pass\n');e=tmp_path/'runs/gaussian-campaign-v01/p2-gravity-refinement-v01/summary.json';e.parent.mkdir(parents=True);e.write_text(json.dumps({'completed':True,'passed':True}))
    path=tmp_path/'protocol.json';r.prepare(path,tmp_path/'out')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'2')
    p=r.validate(path,r.sha(path));assert p['domain_or_image_cohort_admitted'] is False
    with pytest.raises(ValueError,match='declared case'):r.worker(p,path,'arbitrary_extra_case')
    assert not (tmp_path/'out').exists()
    p['cases'][0]['young_Pa']=50000;r.write(path,p)
    with pytest.raises(ValueError,match='Exact14'):r.validate(path,r.sha(path))


def test_controller_retains_spatial_fail_but_stops_on_technical_failure_without_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);protocol=tmp_path/'protocol.json';protocol.write_text('{}');seen=[]
    p={'output':'out','cases':r.cases(),'resources':{'global_wall_cap_s':3600,'per_case_wall_cap_s':600,'physical_GPU_UUID':r.GPU_UUID}}
    monkeypatch.setattr(r,'guard_resources',lambda *_:None)
    class Child:
        def __init__(self,command,**kwargs):
            seen.append(command);self.pid=900000+len(seen);self.returncode=0 if len(seen)==1 else 1;name=command[-1];out=tmp_path/'out'/name;out.mkdir();r.write(out/'result.json',{'completed':len(seen)==1,'passed':False})
        def wait(self,timeout=None):return self.returncode
    monkeypatch.setattr(r.subprocess,'Popen',Child)
    result=r.run(p,protocol)
    assert not result['completed'] and not result['passed'] and len(seen)==2
    assert len(result['queued_case_ids'])==12 and result['retries']==0
    assert result['records'][0]['completed'] and result['records'][0]['passed'] is False
    assert result['records'][1]['exit_code']==1


def test_owned_cleanup_reaps_exit_race_and_uses_only_owned_group(monkeypatch):
    calls=[]
    class Child:
        pid=999123
        def poll(self):return None
        def wait(self,timeout=None):calls.append(('wait',timeout));return 0
    def disappeared(pid,sig):calls.append(('killpg',pid,sig));raise ProcessLookupError()
    monkeypatch.setattr(r.os,'killpg',disappeared);r.stop_owned(Child())
    assert calls[0][0:2]==('killpg',999123) and calls[1]==('wait',3)
