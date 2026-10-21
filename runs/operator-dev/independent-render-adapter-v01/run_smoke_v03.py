"""Single predeclared four-image TRAIN render-only smoke; no bulk capture."""
import datetime,hashlib,json,os,subprocess,xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
ROOT=Path('/home/bayespc2/vscode/guassianSDE');META=ROOT/'runs/operator-dev/independent-render-adapter-v01'
SOURCE=ROOT/'experiments/render_independent_trajectory.py';SOURCE_SHA='da1a2d51d1b7ba7bcb335dafabbec6929f4228c41c749a3fe5297f542e8217d2'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));gpu=next(g for g in xml.findall('gpu') if int(g.findtext('minor_number'))==2);active=[{'pid':p.findtext('pid'),'type':p.findtext('type'),'name':p.findtext('process_name'),'used_memory':p.findtext('used_memory')} for p in gpu.findall('./processes/process_info')]
blocking=[p for p in active if not (p['type']=='G' and p['name']=='/usr/lib/xorg/Xorg' and p['used_memory']=='4 MiB')]
assert not blocking,blocking
query=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();assert int(query.split(',')[1].strip())<256,query
output=ROOT/'runs/gaussian-campaign-v01/independent-render-smoke-v01/t0000-v03';output.parent.mkdir(exist_ok=True);assert not output.exists()
command=['/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python','-u',str(SOURCE),'--input',str(ROOT/'runs/gaussian-campaign-v01/independent-dynamic-gate-v01/expanded-matrix-v01/t0000-temporal-diagonal-hz480'),'--expected-trajectory-sha256','35ce33482af2a7b2fa8bedf94b37d903ddaaaa8e0745072ac1c901424ec436aa','--expected-source-sha256',SOURCE_SHA,'--output',str(output),'--frame-indices','0','15','--candidate-offset-deg','60','10','--resolution','320','240','--gpu','2','--warmup-frames','30','--capture-attempts','8']
env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env.update(OMNI_KIT_ACCEPT_EULA='YES',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
record={'command':command,'source_sha256':SOURCE_SHA,'driver_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU2_prelaunch':query,'gpu2_processes':active,'CUDA_VISIBLE_DEVICES':None,'terminal':False,'completed':False,'output':str(output)}
path=META/'smoke-execution-v03.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
with (META/'smoke-v03.log').open('x') as log:
 process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);record['child_pid']=process.pid;path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'child_pid':process.pid,'output':str(output)}),flush=True);code=process.wait()
try:
 worker=json.loads((output/'execution.json').read_text());summary=json.loads((output/'summary.json').read_text());manifest=json.loads((output/'file-manifest.json').read_text());captures=json.loads((output/'captures.json').read_text())
 assert code==0 and worker['terminal'] and worker['completed'] and worker['completed_images']==4 and summary['completed'] and summary['captured_images']==4
 assert len(captures)==4 and {row['source_frame_index'] for row in captures}=={0,15} and {row['view_id'] for row in captures}=={'primary','candidate00'}
 assert summary['all_sensor_checks_passed'] and summary['maximum_ray_depth_error_m']<=2e-6 and summary['minimum_independent_mask_iou']>=.98
 for rel,sha in manifest.items():assert hashlib.sha256((output/rel).read_bytes()).hexdigest()==sha,rel
 for folder in ['public/observations','private/held_views/candidate00']:
  for channel,extension in [('rgb','png'),('masks','png'),('depth','npz')]:assert {f.stem for f in (output/folder/channel).glob('*.'+extension)}=={'00000','00015'}
 for rel in ['public/observations/cameras.npz','public/query_cameras.npz']:
  with np.load(output/rel,allow_pickle=False) as raw:assert np.array_equal(raw['times_s'],[0.,1.5]) and np.array_equal(raw['source_frame_indices'],[0,15]) and raw['camera_to_world_usd_row'].shape==(2,4,4)
 completed=True;verdict='Worker completed, expected4images/calibration and quantitativechecks passed; manifestverified'
except Exception as error:completed=False;verdict=repr(error)
record.update(terminal=True,completed=completed,exit_code=code,artifact_verdict=verdict,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'terminal':True,'exit_code':code,'completed':completed,'verdict':verdict}),flush=True);raise SystemExit(0 if completed else 1)
