"""Single predeclared four-image TRAIN render-only smoke; no bulk capture."""
import datetime,hashlib,json,os,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('/home/bayespc2/vscode/guassianSDE');META=ROOT/'runs/operator-dev/independent-render-adapter-v01'
SOURCE=ROOT/'experiments/render_independent_trajectory.py';SOURCE_SHA='da45f29c7935c6f3665f550c0cb16e80cb392db05fbffbb64530a7d883fe135b'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));gpu=next(g for g in xml.findall('gpu') if int(g.findtext('minor_number'))==2);active=[{'pid':p.findtext('pid'),'type':p.findtext('type')} for p in gpu.findall('./processes/process_info')]
assert not active,active
query=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();assert int(query.split(',')[1].strip())<256,query
output=ROOT/'runs/gaussian-campaign-v01/independent-render-smoke-v01/t0000-v01';output.parent.mkdir(exist_ok=True);assert not output.exists()
command=['/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python','-u',str(SOURCE),'--input',str(ROOT/'runs/gaussian-campaign-v01/independent-dynamic-gate-v01/expanded-matrix-v01/t0000-temporal-diagonal-hz480'),'--expected-trajectory-sha256','35ce33482af2a7b2fa8bedf94b37d903ddaaaa8e0745072ac1c901424ec436aa','--expected-source-sha256',SOURCE_SHA,'--output',str(output),'--frame-indices','0','15','--candidate-offset-deg','60','10','--resolution','320','240','--gpu','2','--warmup-frames','30','--capture-attempts','8']
env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env.update(OMNI_KIT_ACCEPT_EULA='YES',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
record={'command':command,'source_sha256':SOURCE_SHA,'driver_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU2_prelaunch':query,'gpu2_processes':active,'CUDA_VISIBLE_DEVICES':None,'terminal':False,'completed':False,'output':str(output)}
path=META/'smoke-execution-v01.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
with (META/'smoke-v01.log').open('x') as log:
 process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);record['child_pid']=process.pid;path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'child_pid':process.pid,'output':str(output)}),flush=True);code=process.wait()
record.update(terminal=True,completed=code==0,exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'terminal':True,'exit_code':code}),flush=True);raise SystemExit(code)
