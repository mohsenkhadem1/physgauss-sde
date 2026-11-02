"""Read-only bounded minute-cadence audit monitor; no experiment mutations."""
import subprocess,time,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';script=Path(__file__).with_name('oracle_independent_audit.py');deadline=time.monotonic()+11100
last=None
while time.monotonic()<deadline:
 final=(R/'oracle-reserved-forward-results.json').exists()
 process=subprocess.run([sys.executable,str(script),*([]if final else ['--live'])],cwd=ROOT,capture_output=True,text=True,timeout=120)
 try:state=json.loads(process.stdout)
 except Exception:state=dict(status='monitor-error',stdout=process.stdout,stderr=process.stderr,code=process.returncode)
 if not final:
  p=R/'oracle-independent-live-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(**state,checked_unix=time.time()),indent=2)+'\n');tmp.replace(p)
 current=(state.get('status'),state.get('present'),state.get('failed'))
 if current!=last:print(json.dumps(state),flush=True);last=current
 if final or process.returncode:raise SystemExit(process.returncode)
 time.sleep(min(60,max(0,deadline-time.monotonic())))
print('Audit monitor deadline elapsed; no final experiment report available.',flush=True)
raise SystemExit(2)
