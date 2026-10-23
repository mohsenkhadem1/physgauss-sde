"""Synthetic controls only. Never call fixture posterior or its forward solver."""
import copy,json
import numpy as np
import pytest
from experiments import campaign_compliant_count_continuation_v01 as m
from experiments import campaign_compliant_ridge_reference as old
from experiments import run_compliant_count_continuation_v01 as driver


def synthetic_previous():
 rng=np.random.default_rng(91);z=rng.normal(size=(3000,2));pilot=.5+z@np.array([[.09,0.],[.08,.04]]).T;pilot=pilot[((pilot>0)&(pilot<1)).all(1)]
 def ll(x):return -12*((x-.5)**2).sum(1)
 chain,report=old.ridge_mh(ll,pilot,seed=2026121110,warmup=500,draws=2000)
 return ll,pilot,dict(unit=chain,MH=report)


def test_exact_same_stream_split_vs_uninterrupted_no_reinitialization():
 ll,pilot,previous=synthetic_previous();calls=[];checkpoints=[]
 def guarded_ll(x):
  assert checkpoints;calls.append(len(x));return ll(x)
 chain,report=m.continue_mh(guarded_ll,previous,additional_draws=14000,checkpoint=checkpoints.append)
 uninterrupted,full=old.ridge_mh(ll,pilot,seed=2026121110,warmup=500,draws=16000)
 np.testing.assert_array_equal(chain,uninterrupted);np.testing.assert_array_equal(report['log_likelihood'],full['log_likelihood'])
 for key in ('proposed','accepted','outside_rejected'):np.testing.assert_array_equal(report[key],full[key])
 np.testing.assert_array_equal(chain[:,:2000],previous['unit']);np.testing.assert_array_equal(checkpoints[0]['unit'],previous['unit'][:,-1]);np.testing.assert_array_equal(checkpoints[0]['log_likelihood'],previous['MH']['log_likelihood'][:,-1])
 assert calls and len(checkpoints)==1 and report['continuation']['new_warmup']==0 and report['continuation']['new_seed']is False
 recovered=m.recover_rng(2026121110,previous['MH']['initial_states'],16500)
 assert recovered.bit_generator.state==report['continuation']['RNG_end']


def test_rng_recovery_has_no_posterior_calls_rejects_changed_initial_and_counts():
 ll,pilot,previous=synthetic_previous();initial=previous['MH']['initial_states'];state=m.recover_rng(2026121110,initial,2500).bit_generator.state
 assert state==m.recover_rng(2026121110,initial,2500).bit_generator.state
 changed=initial.copy();changed[0,0]+=1e-12
 with pytest.raises(ValueError,match='initial'):m.recover_rng(2026121110,changed,2500)
 with pytest.raises(ValueError,match='fixed additional'):m.continue_mh(ll,previous,additional_draws=12000)
 bad=copy.deepcopy(previous);bad['MH']['rw_scale']=.4
 with pytest.raises(ValueError,match='proposal'):m.continue_mh(ll,bad,additional_draws=14000)


def test_quantile_MCSE_delta_method_all_chain_denominator_and_long_gates():
 x=np.random.default_rng(45).uniform(size=(8,16000,2));r=m.cdf_quantile_mc(x);extra=m.long_batch_mc(x)
 assert extra['passed'] and extra['time_batch_lengths']==[800,1600]
 flat=x.reshape(-1,2)
 for j in range(2):
  for qi,q in enumerate(np.quantile(flat,[.05,.5,.95],axis=0).T[j]):
   se=[]
   for k in (100,200,400):
    b=(x[:,:,j]<=q).reshape(8,16000//k,k).mean(2).reshape(-1);se.append(b.std(ddof=1)/np.sqrt(8*16000/k))
   lo=max(0,q-.01);hi=min(1,q+.01);density=((flat[:,j]>lo)&(flat[:,j]<=hi)).mean()/(hi-lo)
   assert r['quantile_MCSE'][j][qi]==max(se)/density
 for j in range(2):
  for k in (800,1600):
   b=(x[:,:,j,None]<=np.linspace(0,1,33)).reshape(8,16000//k,k,33).mean(2).reshape(-1,33)
   assert np.all(extra['CDF_batch_MCSE'][j]>=b.std(0,ddof=1)/np.sqrt(len(b))-1e-15)


def test_original_full4D_seeds_prior_and_gates_unchanged_and_only_precision_extended():
 assert driver.SEEDS==(2026121101,2026121102,2026121103)
 assert driver.COUNTS==(8192,32768)
 assert driver.DESIGN['gates']==driver.predecessor.DESIGN['gates']
 assert driver.DESIGN['MH']['full_seed']==2026121120
 assert driver.DESIGN['MH']['qualification_seed']==2026121110
 assert driver.DESIGN['MH']['qualification_draws']==16000
 assert driver.DESIGN['continuation']['new_warmup']==0 and driver.DESIGN['continuation']['no_optional_stopping']
 assert driver.DESIGN['same_original_unit_uniform_intrinsic_prior']


def test_numerical_2D_negative_blocks_all_full4D_and_finalizes_lifecycle(tmp_path,monkeypatch):
 monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver,'validate',lambda*a:dict(output='ref'));monkeypatch.setattr(driver.bundle_io,'guard',lambda*a:None);jobs=[]
 def owned(command,indices,slots,stage,**kw):
  jobs.append(stage.name);assert json.loads((tmp_path/'ref/active-journal.json').read_text())['status']=='BOOTSTRAP_BEFORE_EXPOSURE';driver.io.atomic(stage/'journal.json',dict(completed=True,pid=99999990,records=[],live=[],all_owned_workers_reaped=True));driver.bundle_io.bundle(tmp_path,tmp_path/'ref/jobs/qualify2D',dict(passed=False,completed=True));return[dict(object_index=0,pid=99999991,reaped=True,exit_code=0)]
 monkeypatch.setattr(driver.owner,'owned_jobs',owned);driver.run('fixture','a'*64)
 s=json.loads((tmp_path/'ref/summary.json').read_text());e=json.loads((tmp_path/'ref/execution.json').read_text());assert jobs==['qualification']and s['full4D_started']is False and e['completed']and e['all_owned_workers_reaped']and s['blocked_stages']['six_SMC']=='BLOCKED_BY_2D_GATE'


def test_technical_failure_keeps_active_journal_and_partial_owned_records(tmp_path,monkeypatch):
 monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver,'validate',lambda*a:dict(output='ref'));monkeypatch.setattr(driver.bundle_io,'guard',lambda*a:None)
 def failed(command,indices,slots,stage,**kw):
  assert json.loads((tmp_path/'ref/active-journal.json').read_text())['status']=='BOOTSTRAP_BEFORE_EXPOSURE'
  driver.io.atomic(stage/'journal.json',dict(completed=False,pid=99999990,records=[dict(pid=99999991,reaped=True,exit_code=-15)],live=[],all_owned_workers_reaped=True));raise RuntimeError('Synthetic interrupted owned job')
 monkeypatch.setattr(driver.owner,'owned_jobs',failed)
 with pytest.raises(RuntimeError):driver.run('fixture','a'*64)
 s=json.loads((tmp_path/'ref/failure.json').read_text());r=s['active_stage_journal'];assert r['path']=='ref/qualification/journal.json'and s['all_completed_and_partial_jobs_preserved']
 assert driver.io.checked(tmp_path,r).exists()


def test_long_batch_diagnostic_rejects_persistent_sampling_error():
 rng=np.random.default_rng(89);blocks=rng.uniform(size=(8,8,2));chain=np.repeat(blocks,2000,axis=1)
 assert m.long_batch_mc(chain)['passed']is False
