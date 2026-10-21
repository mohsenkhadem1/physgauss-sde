"""Predeclared tiny gate, then at most one TRAIN replay probe."""
from pathlib import Path
import subprocess,os,json,hashlib,time,xml.etree.ElementTree as ET
ROOT=Path('/home/bayespc2/vscode/guassianSDE');META=ROOT/'runs/operator-dev/cuda-reference-probe-v01';SOURCE=ROOT/'experiments/campaign_cuda_reference_probe.py'
declaration=json.loads((META/'declaration.json').read_text());assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==declaration['source_sha256']
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));g=next(v for v in xml.findall('gpu') if v.findtext('minor_number')=='2');blocking=[v.findtext('pid') for v in g.findall('./processes/process_info') if not (v.findtext('type')=='G' and v.findtext('process_name')=='/usr/lib/xorg/Xorg')];assert not blocking,blocking
root=ROOT/'runs/gaussian-campaign-v01/cuda-reference-probe-v01';root.mkdir(exist_ok=False)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TORCH_DISABLE_NATIVE_JIT':'1'}
ledger={'driver_pid':os.getpid(),'terminal':False,'completed':False,'started_unix_s':time.time(),'source_sha256':declaration['source_sha256'],'jobs':[]}
path=META/'execution.json';assert not path.exists()
def save():path.write_text(json.dumps(ledger,indent=2)+'\n')
save()
for phase in ['controls','train']:
 command=[str(ROOT/'.venv/bin/python'),'-u',str(SOURCE),'--phase',phase,'--output',str(root/phase)]
 if phase=='train':command+=['--reference',str(ROOT/'runs/gaussian-campaign-v01/independent-dynamic-gate-v01/t0001-original-diagonal-hz480'),'--control-proof',str(root/'controls/summary.json')]
 job={'phase':phase,'command':command,'started_unix_s':time.time()};ledger['jobs'].append(job)
 with (META/(phase+'.log')).open('x') as log:
  process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);job['pid']=process.pid;save();print(json.dumps({'phase':phase,'pid':process.pid}),flush=True);code=process.wait()
 job.update(exit_code=code,finished_unix_s=time.time())
 executed=json.loads((root/phase/'execution.json').read_text());job['completed']=code==0 and executed['terminal'] and executed['completed'];save()
 if not job['completed']:break
 summary=json.loads((root/phase/'summary.json').read_text())
 if phase=='controls' and not summary['all_tiny_controls_passed']:break
 if phase=='train' and not summary['parity_passed']:break
ledger.update(terminal=True,completed=len(ledger['jobs'])==2 and all(j['completed'] for j in ledger['jobs']),finished_unix_s=time.time());save();print(json.dumps({'terminal':True,'completed':ledger['completed']}),flush=True);raise SystemExit(0 if ledger['completed'] else 1)
