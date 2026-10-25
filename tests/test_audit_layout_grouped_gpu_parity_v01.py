import copy
import pytest
from experiments import audit_layout_grouped_gpu_parity_v01 as a

def fixture(monkeypatch):
    monkeypatch.setattr(a.run,'record',lambda path:{'path':'release.json','sha256':'bound'})
    tol=dict(risk_absolute_m2=1e-20,risk_relative=1e-10,predictive_mass=2e-12,bank_logZ=1e-8,bank_layout_probability=1e-10,conditional_log_density=1e-10)
    p=dict(devices=['GPU0','GPU1','GPU2'],tolerances=tol,resources=dict(torch_allocated_bytes=1000,aggregate_host_RSS_bytes=2000,worker_wall_s=120))
    saved=[dict(expected_target_variance=1e-10,prior_target_variance=2e-10,predictive_mass=1.,outcome_nodes=n)for n in(8,16)]
    row=dict(slot=0,pid=123,physical_UUID='GPU0')
    v=dict(row,completed=True,qualified=True,protocol_SHA256=a.EXPECTED_PROTOCOL,release={'path':'release.json','sha256':'bound'},oneaction_only=True,all8_ranks_pending=True,no_new_RNG=True,no_new_observation=True,no_reference_requalification=True,bank_replay_errors=dict(logZ=0.,layout_probability=0.),conditional_parity=dict(passed=True,risk_error_m2=0.,log_density_error=0.),levels=[dict(order=n,parity=a.run.compare(s,s,tol),result=s.copy())for n,s in zip((8,16),saved)],torch_peak_allocated_bytes=100,peak_RSS_bytes=500,wall_s=50.,utility_wall_s={'8':1.,'16':2.})
    return row,v,p,saved

def test_recomputes_scalar_parity_not_reported_flag(monkeypatch):
    row,v,p,saved=fixture(monkeypatch);assert len(a.check_device(row,v,p,saved))==2
    v['levels'][0]['result']['expected_target_variance']+=1e-12
    with pytest.raises(ValueError,match='Recomputed'):a.check_device(row,v,p,saved)

def test_rejects_missing_order_wrong_device_and_scope(monkeypatch):
    row,v,p,saved=fixture(monkeypatch)
    for mutate in(lambda x:x['levels'].pop(),lambda x:x.update(physical_UUID='GPU3'),lambda x:x.update(all8_ranks_pending=False)):
        changed=copy.deepcopy(v);mutate(changed)
        with pytest.raises(ValueError):a.check_device(row,changed,p,saved)

def test_memory_and_conditional_error_fail_closed(monkeypatch):
    row,v,p,saved=fixture(monkeypatch)
    for mutate in(lambda x:x.update(torch_peak_allocated_bytes=1001),lambda x:x['conditional_parity'].update(log_density_error=1e-5),lambda x:x['bank_replay_errors'].update(logZ=float('nan'))):
        changed=copy.deepcopy(v);mutate(changed)
        with pytest.raises(ValueError):a.check_device(row,changed,p,saved)
