"""Synthetic kernel/MC/algebra/lifecycle controls only; no fixture posterior runs."""
from pathlib import Path
import copy,json
import numpy as np
import pytest
from experiments import campaign_compliant_ridge_reference as m
from experiments import run_compliant_ridge_reference as driver


def test_fixed_correlated_rejection_kernel_recovers_uniform_with_biased_pilot():
 rng=np.random.default_rng(91);z=rng.normal(size=(3000,2));L=np.array([[.09,0.],[.08,.04]]);pilot=np.clip(np.array([.3,.4])+z@L.T,.01,.99);frozen=[]
 def freeze(mean,cov):frozen.append((mean.copy(),cov.copy()));return dict(synthetic=True)
 def ll(x):assert frozen;return np.zeros(len(x))
 chain,r=m.ridge_mh(ll,pilot,seed=19,warmup=500,draws=3200,freeze_proposal=freeze)
 assert chain.shape==(8,3200,2)and np.all(chain>=0)and np.all(chain<=1)
 assert np.max(abs(chain.mean((0,1))-.5))<.03 and np.max(abs(chain.var((0,1))-1/12))<.012
 assert r['coordinate_reflection']is False and r['correlated_RW_outside_box_rejected']and r['outside_rejected'].sum()>0
 np.testing.assert_array_equal(r['proposal_covariance'],frozen[0][1]);assert len(frozen)==1


def test_correlated_gaussian_target_and_fixed_covariance_degeneracy_failure():
 rng=np.random.default_rng(4);mu=np.array([.5,.5]);cov=np.array([[.0144,.01296],[.01296,.0144]]);L=np.linalg.cholesky(cov);raw=mu+rng.normal(size=(6000,2))@L.T;pilot=raw[((raw>0)&(raw<1)).all(1)];inv=np.linalg.inv(cov)
 ll=lambda x:-.5*np.einsum('bi,ij,bj->b',x-mu,inv,x-mu)
 chain,r=m.ridge_mh(ll,pilot,seed=31,warmup=400,draws=3200);assert np.max(abs(chain.mean((0,1))-mu))<.012
 assert np.max(abs(np.cov(chain.reshape(-1,2),rowvar=False)-cov))<.0015
 assert r['diagnostics']['all_monitored_quantities_pass']and r['ridge_diagnostics']['all_monitored_quantities_pass']
 with pytest.raises(np.linalg.LinAlgError):m.ridge_mh(ll,np.ones((50,2))*.5,seed=31,warmup=1,draws=1)


def test_MCSE_uses_maximum_aligned_batches_and_all_chain_denominator():
 rng=np.random.default_rng(23);x=rng.uniform(size=(8,2000,2));r=m.cdf_quantile_mc(x);assert r['unit_CDF'].shape==(2,33)and r['CDF_batch_MCSE'].shape==(2,33)
 for j in range(2):
  for k in (100,200,400):
   b=(x[:,:,j,None]<=np.linspace(0,1,33)).reshape(8,2000//k,k,33).mean(2).reshape(-1,33);se=b.std(0,ddof=1)/np.sqrt(len(b));assert np.all(r['CDF_batch_MCSE'][j]>=se-1e-15)
 assert all(q is not None for row in r['quantile_MCSE']for q in row)
 with pytest.raises(ValueError):m.cdf_quantile_mc(x[:,:1900])


def test_streaming_prediction_positive_variance_batch_MCSE_and_fixture_functions():
 class Fake:
  X=np.array([[0.,0.,0.],[.1,0,0],[0,.1,0],[0,0,.1],[.1,.1,.1]]);areas=np.ones(5)
  def __call__(self,theta,program):
   value=(theta[:,0]-4000.)*1e-7+(theta[:,3]-15)*1e-6
   if program=='B':value*=2
   return np.broadcast_to(value[:,None,None,None],(len(value),61,5,3)).copy()
 rng=np.random.default_rng(27);x=rng.uniform(size=(8,2000,2));r=m.predictions(Fake(),x,(0,3),chain=True);theta=m.base.physical(x.reshape(-1,2),(0,3));u=Fake()(theta,'A')
 assert np.max(abs(r['A']['mean']-u.mean(0)))<1e-16 and np.max(abs(r['A']['std']-u.std(0)))<1e-16
 assert np.all(r['A']['std']>=0)and np.isfinite(r['A']['mean_batch_MCSE']).all();assert r['known_fixed_functions']==['center','width','fixture_centroid_x','fixture_centroid_y','fixture_centroid_z']
 assert r['function_batch_mean_MCSE'].shape==(18,)and len(r['function_unknown_indices'])==13


def test_qualification_and_pilot_stage_barriers_reject_missing_or_unreaped(tmp_path,monkeypatch):
 monkeypatch.setattr(driver,'ROOT',tmp_path);p=dict(output='reference');out=tmp_path/'reference';out.mkdir();value=dict(completed=True,job='qualify2D',passed=True,protocol=dict(sha256='a'*64));files=driver.bundle_io.bundle(tmp_path,out/'jobs'/'qualify2D',value);row=dict(job='qualify2D',pid=99999999,exit_code=0,reaped=True,files=files);seal=dict(completed=True,protocol_sha256='a'*64,records=[row]);driver.io.write_new(out/'qualification-seal.json',seal)
 driver.stage_barrier(p,'a'*64,'qualification');bad=copy.deepcopy(seal);bad['records'][0]['reaped']=False;driver.io.atomic(out/'qualification-seal.json',bad)
 with pytest.raises(ValueError,match='absent/reaped'):driver.stage_barrier(p,'a'*64,'qualification')
 bad=copy.deepcopy(seal);bad['protocol_sha256']='b'*64;driver.io.atomic(out/'qualification-seal.json',bad)
 with pytest.raises(ValueError):driver.stage_barrier(p,'a'*64,'qualification')
 driver.io.atomic(out/'qualification-seal.json',seal);(out/'jobs/qualify2D/posterior-arrays.npz').unlink()
 with pytest.raises(FileNotFoundError):driver.stage_barrier(p,'a'*64,'qualification')


def test_same_original_sources_prior_seeds_and_no_reflection_or_new_observation():
 frozen=json.loads(Path('runs/operator-dev/compliant-smc-resolution-v04/protocol.json').read_text())
 import hashlib
 for name,h in frozen['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
 assert driver.SEEDS==(2026121101,2026121102,2026121103)and driver.COUNTS==(8192,32768)
 assert driver.DESIGN['new_observations']is False and driver.DESIGN['MH']['proposal_fixed_after_pilot']
 assert driver.DESIGN['gates']['CDF']==.05 and driver.DESIGN['gates']['unit_quantile']==.02 and driver.DESIGN['gates']['bulk_tail_ESS']==400


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
