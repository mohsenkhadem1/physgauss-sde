from pathlib import Path
import hashlib,json,time,os
ROOT=Path.cwd();BASE=ROOT/'runs/gaussian-campaign-v01/c38-nonlinear-table-v01';OUT=ROOT/'runs/operator-dev/completed-metadata-cache-v01';INDEX=ROOT/'runs/operator-dev/c38-nonlinear-table-v01/terminal-case-cache.json';OUT.mkdir(parents=True,exist_ok=True);records={}
while True:
 terminal=[]
 for i in range(3):
  execution=BASE/f'gpu{i}/execution.json'
  if not execution.exists():terminal.append(False);continue
  try:d=json.loads(execution.read_text())
  except json.JSONDecodeError:terminal.append(False);continue
  terminal.append(d.get('terminal',False))
  for row in d.get('records',[]):
   if not row.get('terminal') or not row.get('result'):continue
   rec=row['result'];src=ROOT/rec['path'];data=src.read_bytes();digest=hashlib.sha256(data).hexdigest()
   if digest!=rec['sha256']:raise ValueError('Terminal case changed after controller hash')
   dest=OUT/(digest+'.json')
   if dest.exists():assert dest.read_bytes()==data
   else:
    with dest.open('xb') as f:f.write(data)
   records[rec['path']]=dict(path=rec['path'],sha256=digest,cache=str(dest.relative_to(ROOT)),bytes=len(data),phase=i,completed=row['completed'],passed=row['passed'])
 result=dict(monitor_pid=os.getpid(),all_phases_terminal=all(terminal),records=list(records.values()),cached_only_after_worker_terminal=True)
 temp=INDEX.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(INDEX)
 if all(terminal):break
 time.sleep(5)
print(json.dumps(dict(terminal=True,cached=len(records))))
