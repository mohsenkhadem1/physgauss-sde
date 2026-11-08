from pathlib import Path
import subprocess,datetime,os,json,time
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop';deadline=datetime.datetime.fromisoformat('2026-11-07T21:44:00+00:00').timestamp();state=dict(pid=os.getpid(),status='waiting_for_native_campaign');dest=REPORT/'illustration-status.json'
def save():state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dest.write_text(json.dumps(state,indent=2))
while time.time()<deadline:
 d=json.loads((REPORT/'supervisor.json').read_text());save()
 if d['status'] in ['complete','finished_with_failures']:break
 time.sleep(20)
else:state['status']='deadline_incomplete';save();raise SystemExit(1)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4');state['status']='rendering';save()
for script,args in [('drop_render_illustration.py',[]),('drop_visualize.py',['--wide'])]:
 with (BASE/(script+'-wide.log')).open('w') as log:p=subprocess.run([str(BASE/'env/bin/python'),'scripts/gap_first_v01/'+script,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:state.update(status='failed',script=script,returncode=p.returncode);save();raise SystemExit(1)
state['status']='complete';save()
