from pathlib import Path
import json,hashlib,time,os
ROOT=Path.cwd();META=ROOT/'runs/operator-dev/p2-contact-spatial-pairs-v01';OUT=ROOT/'runs/gaussian-campaign-v01/p2-contact-spatial-pairs-v01';CACHE=ROOT/'runs/operator-dev/completed-metadata-cache-v01';records={}
while True:
 terminal=[]
 for phase in ['324tets','768tets']:
  f=META/phase/'execution.json'
  try:d=json.loads(f.read_text())
  except (FileNotFoundError,json.JSONDecodeError):terminal.append(False);continue
  terminal.append(d.get('terminal',False))
  for row in d['workers']:
   if not row.get('terminal'):continue
   src=OUT/phase/row['case']/'summary.json'
   if not src.exists():continue
   data=src.read_bytes();h=hashlib.sha256(data).hexdigest();dest=CACHE/(h+'.json')
   if dest.exists():assert dest.read_bytes()==data
   else:
    with dest.open('xb') as stream:stream.write(data)
   records[str(src.relative_to(ROOT))]=dict(path=str(src.relative_to(ROOT)),sha256=h,cache=str(dest.relative_to(ROOT)),bytes=len(data),phase=phase,worker_reaped=True)
 result=dict(monitor_pid=os.getpid(),all_phases_terminal=all(terminal),records=list(records.values()));f=META/'terminal-case-cache.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(f)
 if all(terminal):break
 time.sleep(5)
print(json.dumps(dict(completed=True,cached=len(records))))
