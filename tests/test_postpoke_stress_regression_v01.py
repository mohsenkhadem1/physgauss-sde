import copy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments import campaign_postpoke_stress_regression_v01 as h
from experiments import run_postpoke_stress_regression_v01 as run


def grid():
 seeds=[[2026200000+100*i+j for j in range(16)]for i in range(12)];rows=[]
 bad={2:[(0,3),(2,7),(5,12)],4:[(4,11)],8:[(1,2),(3,9),(5,15)],10:[(2,1),(4,10)]}
 for obj in range(12):
  rows.append([dict(action=a,noise_seed=seed,importance=dict(importance_gate_passed=(a,j)not in bad.get(obj,[])))for a in range(6)for j,seed in enumerate(seeds[obj])])
 return rows,seeds


def test_complete_membership_includes_all_nine_failures_not_selected_noise0():
 rows,seeds=grid();cases=h.failed_cases(rows,seeds)
 assert len(cases)==9 and [r['case_index']for r in cases]==list(range(9))
 assert all(r['noise_index']!=0 for r in cases)
 permuted=[list(reversed(r))for r in rows]
 assert h.failed_cases(permuted,seeds)==cases
 for change in ['missing','duplicate','hidefailure']:
  changed=copy.deepcopy(rows)
  if change=='missing':changed[0].pop()
  elif change=='duplicate':changed[0][-1]=changed[0][0]
  else:changed[2][3]['importance']['importance_gate_passed']=True
  with pytest.raises(ValueError):h.failed_cases(changed,seeds)


def test_arbitrary_action_and_noise_replay_uses_correct_cached_slot():
 old=h.old.model.cohort.old;rng=np.random.default_rng(3);N=32;target=rng.normal(size=(N,4))*1e-4;pokes=rng.normal(size=(6,N,12))*1e-5
 case=dict(action=4,noise_index=7,noise_seed=2026200207)
 observed=np.random.default_rng(case['noise_seed']).normal(0,old.original.SIGMA,12)
 w,_=old.exact_reweight(pokes[4],observed);mu,var=old.weighted_prediction(target,w)
 means=np.zeros((96,4));variances=means.copy();slot=4*16+7;means[slot]=mu;variances[slot]=var
 class FakeJoint:
  def __init__(self,force,a,b):pass
  def predictions(self,x):return np.zeros((len(x),2,12))
 fake=NS(JointObservationForward=FakeJoint,cohort=NS(old=old))
 result=h.replay_observation(fake,np.zeros(12),np.ones(4)*.5,case,target,pokes,means,variances)
 np.testing.assert_array_equal(result[0],observed);assert result[2]['noise_index']==7
 np.testing.assert_allclose(result[3],mu);assert not result[2]['old_importance_diagnostic']['importance_gate_passed']
 # A wrongnoise0 cachelookup fails; observation remains unchanged.
 with pytest.raises(ValueError,match='replay failed'):
  h.replay_observation(fake,np.zeros(12),np.ones(4)*.5,dict(case,noise_index=0),target,pokes,means,variances)


def terminal():
 e=dict(completed=True,all_owned_workers_reaped=True,protocol_sha256='frozen',pid=2,records=[dict(object_index=i,pid=10+i,exit_code=0,reaped=True)for i in range(12)])
 r=dict(completed=True,exit_code=0,controller_reaped=True,controller_PID_absent=True,all_observed_owned_worker_PIDs_absent=True,controller_pid=2,supervisor_pid=1,observed_owned_worker_PIDs=list(range(10,22)))
 s=dict(completed=True,objects=12,all12_retained=True,all_numerical_gates_passed=False)
 return e,r,s


def test_count_barrier_retains_numerical_failures_but_rejects_live_partial():
 e,r,s=terminal();calls=[];assert h.terminal_count_gate(e,r,s,'frozen',calls.append)
 assert calls==[1,2]+list(range(10,22))
 for change in ['partial','unreaped','mismatch','digest','nonzeroexit']:
  e,r,s=terminal()
  if change=='partial':e['records'].pop()
  if change=='unreaped':e['records'][0]['reaped']=False
  if change=='mismatch':r['observed_owned_worker_PIDs'][0]=999
  if change=='digest':e['protocol_sha256']='bad'
  if change=='nonzeroexit':r['exit_code']=1
  with pytest.raises(ValueError):h.terminal_count_gate(e,r,s,'frozen',lambda pid:None)
 e,r,s=terminal()
 with pytest.raises(ValueError,match='live'):
  h.terminal_count_gate(e,r,s,'frozen',lambda pid:(_ for _ in ()).throw(ValueError('live owned PID/group')))


def test_seeds_counts_resources_and_gates_frozen_before_outcomes():
 assert run.DESIGN['particles']==32768 and run.DESIGN['MH']['draws']==64000 and run.DESIGN['MH']['prior_initialization']
 assert run.DESIGN['gates']==run.old.DESIGN['gates']
 assert 'all32768' in run.DESIGN['bridge']['initialization']
 assert all(v==run.old.DESIGN['bridge'][k]for k,v in run.DESIGN['bridge'].items()if k!='initialization')
 assert run.RESOURCES['worker_slots']==[[i]for i in range(9)]and run.RESOURCES['controller_cores']==[16]
 assert run.RESOURCES['worker_RSS_bytes']==2*1024**3 and run.RESOURCES['worker_wall_s']==2400 and run.RESOURCES['global_wall_s']==3000
 seeds=sum(run.DESIGN['fresh_seeds'],[])+sum(run.DESIGN['bridge_seeds'],[])+sum(run.DESIGN['permutation_seeds'],[])+run.DESIGN['MH_seeds']
 assert len(seeds)==len(set(seeds))==90


def test_regression_aggregate_preserves_all_failed_results_and_never_claims_policy():
 rows=[dict(index=i,completed=True,old_importance_failed=True,observation_replay_passed=True,passed=i!=7)for i in range(9)]
 r=run.aggregate(rows);assert r['cases']==9 and r['numerically_passed_cases']==8 and not r['all_numerical_gates_passed']and not r['policy_advantage_claim']
 with pytest.raises(ValueError):run.aggregate(rows[:-1])


def test_generated_worker_and_controller_commands_pass_real_child_guards():
 import subprocess
 commands=[run.worker_command('unused','unused',i,i,guard_only=True)for i in range(9)]+[run.controller_command('unused','unused',guard_only=True)]
 for command in commands:
  child=subprocess.run(command,capture_output=True,text=True,timeout=60)
  assert child.returncode==0,child.stderr


def test_generated_child_rejects_wrong_threads_affinity_and_visible_GPU():
 import subprocess
 original=run.worker_command('unused','unused',0,0,guard_only=True)
 for change in ('threads','affinity','GPU'):
  command=original.copy()
  if change=='threads':command=[s.replace('_NUM_THREADS=1','_NUM_THREADS=2')for s in command]
  elif change=='affinity':command[command.index('-c')+1]='15'
  else:command[command.index('CUDA_VISIBLE_DEVICES=')]='CUDA_VISIBLE_DEVICES=0'
  child=subprocess.run(command,capture_output=True,text=True,timeout=60)
  assert child.returncode!=0
  assert ('one-thread'in child.stderr if change=='threads' else 'affinity'in child.stderr if change=='affinity' else 'no visible GPU'in child.stderr)
