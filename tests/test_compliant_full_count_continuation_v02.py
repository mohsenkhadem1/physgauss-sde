"""Synthetic stream/memory/MCSE/lifecycle controls; no fixture posterior calls."""
import json,resource,time
import numpy as np
import pytest
from experiments import campaign_compliant_full_count_continuation_v01 as m
from experiments import campaign_compliant_ridge_reference as old
from experiments import run_compliant_full_count_continuation_v02 as driver


def test_exact64000_split_stream_states_likelihood_counts_and_final_RNG():
 rng=np.random.default_rng(19);L=np.array([[.08,0,0,0],[.06,.06,0,0],[.03,.02,.07,0],[.04,.01,.03,.06]]);pilot=.5+rng.normal(size=(4000,4))@L.T;pilot=pilot[((pilot>0)&(pilot<1)).all(1)]
 def ll(x):return -12*((x-.5)**2).sum(1)
 first,report=old.ridge_mh(ll,pilot,seed=2026121120,warmup=1000,draws=4000);previous=dict(unit=first,MH=report);checkpoints=[]
 def guarded(x):assert checkpoints;return ll(x)
 continued,result=m.continue_mh(guarded,previous,additional_draws=60000,checkpoint=checkpoints.append)
 uninterrupted,full=old.ridge_mh(ll,pilot,seed=2026121120,warmup=1000,draws=64000)
 np.testing.assert_array_equal(continued,uninterrupted);np.testing.assert_array_equal(result['log_likelihood'],full['log_likelihood'])
 for k in ('proposed','accepted','outside_rejected'):np.testing.assert_array_equal(result[k],full[k])
 np.testing.assert_array_equal(checkpoints[0]['unit'],first[:,-1]);np.testing.assert_array_equal(continued[:,:4000],first)
 assert result['continuation']['RNG_end']==m.recover_rng(2026121120,report['initial_states'],65000).bit_generator.state
 with pytest.raises(ValueError,match='additional'):m.continue_mh(ll,previous,additional_draws=56000)


class Fake:
 X=np.array([[0.,0.,0.],[.1,0,0],[0,.1,0],[0,0,.1],[.1,.1,.1]]);areas=np.ones(5)
 def __call__(self,theta,program):
  value=(theta[:,0]-4000.)*1e-7+(theta[:,3]-15)*1e-6+(theta[:,1]-.04)*1e-5
  if program=='B':value*=2
  return np.broadcast_to(value[:,None,None,None],(len(value),61,5,3)).copy()


def test_streaming_prediction_exact_moments_and_MCSE_against_direct_small_control():
 x=np.random.default_rng(29).uniform(size=(8,2000,4));r=m.predictions(Fake(),x,(0,1,2,3),chain=True);theta=m.base.physical(x.reshape(-1,4),(0,1,2,3));u=Fake()(theta,'A');mean=u.mean(0);std=u.std(0)
 np.testing.assert_allclose(r['A']['mean'],mean,rtol=0,atol=1e-16);np.testing.assert_allclose(r['A']['std'],std,rtol=0,atol=1e-16)
 se_mean=[];se_std=[]
 for k in (100,200,400):
  batches=u.reshape(8,2000//k,k,61,5,3);bm=batches.mean(2).reshape(-1,61,5,3);bs=((batches-mean)**2).mean(2).reshape(-1,61,5,3)
  se_mean.append(bm.std(0,ddof=1)/np.sqrt(len(bm)));se_std.append(bs.std(0,ddof=1)/np.sqrt(len(bs))/(2*std))
 np.testing.assert_allclose(r['A']['mean_batch_MCSE'],np.max(se_mean,axis=0),rtol=1e-12,atol=1e-16);np.testing.assert_allclose(r['A']['std_delta_batch_MCSE'],np.max(se_std,axis=0),rtol=1e-12,atol=1e-16)


def test_target_size_full_prediction_and_diagnostics_memory_below_cap():
 # Same array dimensions and complete predictor/function/parameter/ridge diagnosis
 # as the future run, using only a synthetic uniform target and cheap fields.
 x=np.random.default_rng(44).uniform(size=(8,64000,4));r=m.predictions(Fake(),x,(0,1,2,3),chain=True);q=m.cdf_quantile_mc(x);long=m.long_batch_mc(x)
 assert r['function_diagnostics']['all_monitored_quantities_pass']and long['passed']
 assert all(v is not None and v<=.005 for row in q['quantile_MCSE']for v in row)
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3


def test_frozen_design_only_extends_fullMH_and_reuses_all_original_stages():
 assert driver.DESIGN['gates']==driver.prior.DESIGN['gates'];assert driver.DESIGN['MH']['total_retained']==64000
 assert driver.DESIGN['MH']['additional_retained']==60000 and driver.DESIGN['MH']['new_warmup']==0
 assert driver.DESIGN['MH']['seed']==2026121120 and driver.DESIGN['MH']['new_seed']is False
 assert driver.DESIGN['reuse_all_six_SMC']and driver.DESIGN['reuse_all_fifteen_SMC_pairwise']and driver.DESIGN['reuse_2D_qualification']
 assert driver.RESOURCES['worker_slots']==[[12,13]]and driver.RESOURCES['worker_RSS_bytes']==2*1024**3


def test_technical_failure_preserves_active_owned_journal_and_does_not_retry(tmp_path,monkeypatch):
 monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver,'validate',lambda*a:dict(output='ref'));monkeypatch.setattr(driver.bundle_io,'guard',lambda*a:None);calls=[]
 def owned(command,indices,slots,stage,**kw):
  calls.append(1);assert json.loads((stage/'journal.json').read_text())['status']=='BOOTSTRAP_BEFORE_EXPOSURE';driver.io.atomic(stage/'journal.json',dict(pid=99999990,records=[dict(pid=99999991,reaped=True,exit_code=-15)],live=[],all_owned_workers_reaped=True));raise RuntimeError('Synthetic interrupted owned job')
 monkeypatch.setattr(driver.owner,'owned_jobs',owned)
 with pytest.raises(RuntimeError):driver.run('fixture','a'*64)
 result=json.loads((tmp_path/'ref/failure.json').read_text());assert result['no_retry']and result['all_completed_and_partial_jobs_preserved']and calls==[1];driver.io.checked(tmp_path,result['active_stage_journal'])


if __name__=='__main__':
 start=time.monotonic();test_target_size_full_prediction_and_diagnostics_memory_below_cap();print(json.dumps(dict(synthetic_full_count_memory_control=True,fixture_posterior_calls=0,wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))


def test_numerical_precision_negative_finishes_once_with_reused_SMC_only(tmp_path,monkeypatch):
 monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver.bundle_io,'guard',lambda*a:None);summary_path=tmp_path/'oldsummary.json';driver.io.write_new(summary_path,dict(SMC_logZ_range=.02,pairwise_comparisons=[dict(passed=True)]*15));p=dict(output='ref',inputs=dict(previous_summary=driver.io.record(tmp_path,summary_path)));monkeypatch.setattr(driver,'validate',lambda*a:p)
 cached=driver.bundle_io.bundle(tmp_path,tmp_path/'cached',dict(synthetic=True));monkeypatch.setattr(driver,'files',lambda p,job:cached);monkeypatch.setattr(driver.model,'compare',lambda*a:dict(passed=True));calls=[]
 def owned(command,indices,slots,stage,**kw):
  calls.append(command(0,0));assert indices==[0]and slots==1 and kw['worker_wall_s']==1200
  value=dict(passed=False,MC_diagnostics_passed=False,long_batch_MC=dict(passed=False,CDF_batch_MCSE=np.array([[.002]]),quantile_MCSE=[[.006]]),CDF_batch_MCSE=np.array([[.002]]),quantile_MCSE=[[.006]],field_MCSE_passed=True,maximum_function_relative_MCSE=.02,old_prefix_bit_identical=True,worker_wall_s=20.,worker_peak_RSS_bytes=1_000_000)
  driver.bundle_io.bundle(tmp_path,tmp_path/'ref/jobs/fullMH',value);return [dict(object_index=0,pid=99999999,reaped=True,exit_code=0)]
 monkeypatch.setattr(driver.owner,'owned_jobs',owned);driver.run('fixture','a'*64);result=json.loads((tmp_path/'ref/summary.json').read_text());execution=json.loads((tmp_path/'ref/execution.json').read_text())
 assert result['completed']and result['passed']is False and result['six_SMC_reused']and len(result['MH_vs_SMC'])==6
 assert execution['all_owned_workers_reaped']and len(calls)==1 and 'experiments.run_compliant_full_count_continuation_v02'in calls[0]


def test_predecessor_identity_barrier_rejects_live_PID_and_group(monkeypatch):
 monkeypatch.setattr(driver,'proc_identity',lambda pid:(1,'stamp'))
 with pytest.raises(ValueError,match='PID'):driver.absent_process_and_group(42)
 monkeypatch.setattr(driver,'proc_identity',lambda pid:None);monkeypatch.setattr(driver.os,'killpg',lambda pid,sig:None)
 with pytest.raises(ValueError,match='group'):driver.absent_process_and_group(42)
 def absent(pid,sig):raise ProcessLookupError()
 monkeypatch.setattr(driver.os,'killpg',absent);assert driver.absent_process_and_group(42)
 def permission_denied(pid,sig):raise PermissionError()
 monkeypatch.setattr(driver.os,'killpg',permission_denied)
 with pytest.raises(PermissionError):driver.absent_process_and_group(42)
