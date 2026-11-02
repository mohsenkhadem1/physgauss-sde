"""Bounded durable waiter for independently released primary unknown-object study."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from scripts.compositional_neural_mechanics_reporting import unknown_reaction_secondary as replay
from scripts.compositional_neural_mechanics.dataset_runner import write

def main():
 replay.guard(False);clock=replay.primary.B/'stage-clock.json';deadline=json.loads(clock.read_text())['deadline_unix']+2400
 pins={str(p.relative_to(replay.ROOT)):replay.sha(p)for p in (Path(__file__).resolve(),replay.PLAN,clock,replay.R/'unknown-reaction-prescore-independent-review.json')};status=replay.R/'unknown-reaction-waiter-status.json'
 while time.time()<deadline:
  if any(replay.sha(replay.ROOT/p)!=h for p,h in pins.items()):raise ValueError('Waiter plan/review/clock changed')
  if replay.AUDIT.exists():
   try:replay.guard(True)
   except Exception as error:
    write(status,dict(state='audit_rejected',error=repr(error),evidence_sha256=pins));return
   if replay.OUT.exists():raise ValueError('Secondary replay already started; no restart')
   log=replay.R/'unknown-reaction-secondary-service.log'
   with log.open('x')as stream:
    process=subprocess.Popen([sys.executable,'-m','scripts.compositional_neural_mechanics_reporting.unknown_reaction_secondary','--run'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=min(1830,max(.01,deadline-time.time())))
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
   write(status,dict(state='complete'if code==0 else 'failed',exit_code=code,evidence_sha256=pins,final_independent_audit_requested=True));return
  write(status,dict(state='waiting_for_passed_primary_audit',deadline_unix=deadline,evidence_sha256=pins));time.sleep(min(30,max(0,deadline-time.time())))
 write(status,dict(state='deadline_expired_without_restart',evidence_sha256=pins))
if __name__=='__main__':main()
