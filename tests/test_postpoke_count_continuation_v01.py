"""Saved-state equality, count-only recipe, and completed-reference barriers."""
import copy
import numpy as np
import pytest
from experiments import campaign_postpoke_count_continuation_v01 as helper
from experiments import run_postpoke_count_continuation_v01 as run


def test_count_only_recipe_preserves_prior_observation_seeds_and_gates():
 assert run.DESIGN['particles']==4*run.previous.DESIGN['particles']==32768
 assert run.DESIGN['objects']==run.previous.DESIGN['objects']and run.DESIGN['new_MH_draws']==0
 assert run.DESIGN['gates']==run.previous.DESIGN['gates']
 assert run.RESOURCES['worker_slots']==[[i]for i in range(12)]and run.RESOURCES['controller_cores']==[16]
 assert run.RESOURCES['threads']==1 and run.RESOURCES['worker_wall_s']==2400 and run.RESOURCES['global_wall_s']==3000
 assert run.DESIGN['same_bridge_fresh_and_subsample_seeds']and run.DESIGN['reuse135_prediction_moments_and_MCSE_bitexact']


def test_function_replay_matches_old_full11RHS_with_arbitrary_chunk_boundaries():
 import torch
 torch.set_num_threads(1)
 u=np.array([[.01,.2,.4,.6],[.99,.6,.1,.3],[.4,.9,.8,.2]])
 f=helper.model.cohort.old.ArbitraryForceForward();theta=helper.model.cohort.physical(u)
 _,_,full=helper.model.cohort.joint_features(f,theta,chunk=2,with_functions=True)
 for chunk in(1,2,4):
  r=helper.replay_function_moments(f,u,chunk=chunk)
  np.testing.assert_allclose(r['function_mean'],full.mean(0),atol=1e-13,rtol=1e-12)
  np.testing.assert_allclose(r['function_std'],full.std(0),atol=1e-13,rtol=1e-12)
  assert r['new_MH_draws']==0 and r['saved_draws']==3


def fixtures():
 e=dict(completed=True,all_owned_workers_reaped=True,protocol_sha256=run.PREVIOUS_DIGEST,pid=2,records=[dict(object_index=i,pid=10+i,exit_code=0,reaped=True)for i in range(12)])
 receipt=dict(completed=True,controller_reaped=True,controller_PID_absent=True,all_observed_owned_worker_PIDs_absent=True,controller_pid=2,supervisor_pid=1,observed_owned_worker_PIDs=list(range(10,22)))
 summary=dict(completed=True,objects=12,all12_retained=True);rows=[dict(completed=True,index=i,MH_passed=True,observation_replay_passed=True)for i in range(12)]
 return e,receipt,summary,rows


def test_terminal_barrier_checks_all_owned_PID_groups_and_all12MH():
 e,r,s,rows=fixtures();calls=[];assert run.terminal_gate(e,r,s,rows,calls.append);assert calls==[1,2]+list(range(10,22))
 rows[9]['MH_passed']=False
 with pytest.raises(ValueError):run.terminal_gate(e,r,s,rows,lambda pid:None)


def test_terminal_barrier_rejects_partial_mismatched_and_live_processes():
 for change in('partial','unreaped','wrongPIDset','wrongprotocol'):
  e,r,s,rows=fixtures()
  if change=='partial':e['records'].pop()
  if change=='unreaped':e['records'][3]['reaped']=False
  if change=='wrongPIDset':r['observed_owned_worker_PIDs'][-1]=200
  if change=='wrongprotocol':e['protocol_sha256']='bad'
  with pytest.raises(ValueError):run.terminal_gate(e,r,s,rows,lambda pid:None)
 e,r,s,rows=fixtures()
 def live(pid):raise ValueError('live PID/group')
 with pytest.raises(ValueError,match='live'):run.terminal_gate(e,r,s,rows,live)


def test_failed_objects_never_dropped_from_count_only_aggregate():
 rows=[dict(completed=True,index=i,passed=i!=4)for i in range(12)];r=run.aggregate(rows)
 assert r['objects']==12 and r['numerically_passed_objects']==11 and not r['all_numerical_gates_passed']and r['new_MH_draws']==0
 with pytest.raises(ValueError):run.aggregate(rows[:-1])


def test_actual_generated12worker_and_controller_commands_match_one_thread_guard():
 import json,subprocess
 children=[]
 for slot in range(12):
  cmd=run.worker_command('CONTROL_ONLY','NONE',slot,slot);cmd[cmd.index('worker')]='guard-worker'
  children.append((slot,subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)))
 for slot,child in children:
  stdout,stderr=child.communicate(timeout=60);assert child.returncode==0,stderr
  row=json.loads(stdout);assert row==dict(CPU=[slot],threads=1,GPU_visible='')
 cmd=run.controller_command('CONTROL_ONLY','NONE');cmd[cmd.index('run')]='guard-controller'
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert result.returncode==0,result.stderr
 assert json.loads(result.stdout)==dict(CPU=[16],threads=1,GPU_visible='')
