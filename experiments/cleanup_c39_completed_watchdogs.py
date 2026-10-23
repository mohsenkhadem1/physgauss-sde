"""Resource-only cleanup of blocked daemon watchdogs AFTER completed strict cases.

Never signals a mechanics worker or a running/partial case. Frozen solver and
controller bytes remain unchanged. Every signalled watchdog is an actual direct
child of a protocol-bound owned worker whose two completed captures hash-match.
"""
import argparse,json,os,signal,time
from pathlib import Path
from experiments import run_campaign_c39_side_table as c
CONTROLLER=2282388;SOURCE='experiments/cleanup_c39_completed_watchdogs.py';OUT=c.OP/'completed-watchdog-cleanup.json'

def once():
 s=json.loads((c.OP/'execution.json').read_text());actions=[]
 if s['controller_pid']!=CONTROLLER:raise ValueError('Exact owned controller only')
 for row in s['records']:
  if row['status']!='RUNNING':continue
  worker=row['pid'];proc=Path('/proc')/str(worker)
  if not proc.exists():continue
  pp=next(x for x in (proc/'status').read_text().splitlines() if x.startswith('PPid:'))
  if int(pp.split()[1])!=CONTROLLER:raise ValueError('Owned worker parent mismatch')
  i=row['phase_index'];protocol=c.OP/('gpu'+str(i))/'protocol-v02.json';p,_=c.validate(str(protocol.relative_to(c.ROOT)),c.sha(protocol));f=c.ROOT/p['output']/row['case']['id']/'result.json'
  if not f.exists():continue
  result=json.loads(f.read_text())
  if not result.get('completed') or not result.get('physics_passed') or len(result.get('meshes',[]))!=2:continue
  if result['pid']!=worker or result['case']!=row['case'] or result['protocol']!=c.record(protocol):raise ValueError('Completed case identity mismatch')
  if any(not m['completed'] or not m['physics_passed'] or c.sha(c.ROOT/m['capture']['path'])!=m['capture']['sha256'] for m in result['meshes']):raise ValueError('Complete captured physical checks required')
  mask=int(next(x for x in (proc/'status').read_text().splitlines() if x.startswith('SigBlk:')).split()[1],16)
  if not mask&(1<<(signal.SIGTERM-1)):continue
  cmd=(proc/'cmdline').read_bytes();children=(proc/'task'/str(worker)/'children').read_text().split()
  if not children:continue
  if len(children)!=1 or os.getpgid(worker)!=worker:raise ValueError('Unique watchdog and owned worker PGID required')
  for child in children:
   cp=Path('/proc')/child
   if not cp.exists():continue
   if (cp/'cmdline').read_bytes()!=cmd:raise ValueError('Unexpected owned child; no signal sent')
   parent=int(next(x for x in (cp/'status').read_text().splitlines() if x.startswith('PPid:')).split()[1])
   if parent!=worker or os.getpgid(int(child))!=worker:raise ValueError('Watchdog direct parent/PGID mismatch')
   child_mask=next(x for x in (cp/'status').read_text().splitlines() if x.startswith('SigBlk:')).split()[1]
   os.kill(int(child),signal.SIGKILL);actions.append(dict(worker_pid=worker,watchdog_pid=int(child),case_id=row['case']['id'],completed_result=c.record(f),signal='SIGKILL watchdog only',worker_never_signalled=True,source_and_capture_hashes_verified=True,worker_PGID=worker,watchdog_PGID=worker,watchdog_PPid=worker,worker_signal_mask_before=hex(mask),watchdog_signal_mask_before=child_mask,capture_records=[m['capture'] for m in result['meshes']]))
 return s,actions
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();begin=time.monotonic();r=dict(completed=False,source=c.record(c.ROOT/SOURCE),controller_pid=CONTROLLER,resource_only=True,actions=[])
 try:
  while True:
   state,actions=once();r['actions'].extend(actions)
   for action in r['actions']:
    matched=next(x for x in state['records'] if x['case']['id']==action['case_id']);action.update(worker_absent_after=not Path('/proc',str(action['worker_pid'])).exists(),watchdog_absent_after=not Path('/proc',str(action['watchdog_pid'])).exists(),worker_controller_status_after=matched['status'],worker_exit_code_after=matched.get('exit_code'))
   r['seconds']=time.monotonic()-begin;c.write(OUT,r)
   if state['terminal'] or not a.watch:r['completed']=True;break
   if time.monotonic()-begin>4400:raise TimeoutError('Bounded resource-only observer')
   time.sleep(1.)
 finally:r['seconds']=time.monotonic()-begin;c.write(OUT,r)
 print(json.dumps(dict(completed=r['completed'],actions=len(r['actions']))))
