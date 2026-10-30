"""Future independent replay of one declared frozen job, never a campaign restart."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,importlib.metadata,json,os,signal,subprocess,sys,time,shutil,platform
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
REPORT=ROOT/'reports/coupled-surface-v13';BASE=ROOT/'runs/coupled-surface-v13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def environment():
 compiler=os.environ.get('CXX') or shutil.which('c++') or shutil.which('g++')
 fallback=Path.home()/'.local/share/physgauss-external/pidg-official-v01/env/bin/x86_64-conda-linux-gnu-c++'
 if compiler is None and fallback.exists():compiler=str(fallback)
 compiler_info=None
 if compiler:
  compiler_info=dict(path=str(Path(compiler).resolve()),version=subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True,timeout=5).stdout)
 return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},compiler=compiler_info)

def verify():
 from scripts.coupled_contact_v13.surface_scope import check_qualification
 f=read(REPORT/'method-freeze.json')
 for rel,h in f['source_sha256'].items():
  path=Path(rel)
  if path.is_absolute() or '..' in path.parts or sha(ROOT/path)!=h:raise RuntimeError('Frozen source mismatch: '+rel)
 if sha(REPORT/'protocol.json')!=f['protocol_sha256']:raise RuntimeError('Protocol changed')
 check_qualification(REPORT,ROOT)
 return f,read(REPORT/'protocol.json')
def declared(protocol):
 jobs=[]
 for family,scales in [('fem',protocol['fem_young_grid']),('native',protocol['native_scale_grid']),('rigid',[6000.])]:
  for i,scale in enumerate(scales):
   for j,mu in enumerate(protocol['friction_grid']):jobs.append(dict(path=f'library/{family}/{i}-{j}',family=family,scenario='squeeze',scale=scale,friction=mu,reference=False))
 for obj in protocol['objects']:
  for scene in ['squeeze']+protocol['forecast_scenarios']:
   job=dict(path=f"reference/{obj['id']}/{scene}",family='fem',scenario=scene,scale=obj['young'],friction=obj['friction'],reference=True)
   jobs.extend([job,dict(job,path=job['path'].replace('reference/','reference-refinement/'),refinement=True),dict(job,path=job['path'].replace('reference/','reference-spatial/'),spatial=True)])
 return jobs

def select_job(selector,protocol):
 matches=[j for j in declared(protocol) if j['path']==selector];evidence={}
 if matches:return matches[0],evidence
 parts=Path(selector).parts
 if len(parts)!=4 or parts[0]!='prediction':raise ValueError('Undeclared job selector')
 _,oid,scene,family=parts
 objects=[o for o in protocol['objects'] if o['id']==oid]
 if len(objects)!=1 or scene not in protocol['forecast_scenarios'] or family not in ('fem','native','rigid','oracle'):raise ValueError('Undeclared prediction')
 if family=='oracle':scale,mu=objects[0]['young'],objects[0]['friction']
 else:
  source=BASE/'inference'/oid/'selection.json';selected=read(source)[family]
  if not selected.get('success'):raise ValueError('No successful declared fit')
  scale,mu=selected['scale'],selected['friction'];evidence[str(source.relative_to(ROOT))]=sha(source)
  scales=protocol['fem_young_grid'] if family=='fem' else protocol['native_scale_grid'] if family=='native' else [6000.]
  if scale not in scales or mu not in protocol['friction_grid']:raise ValueError('Fit outside frozen grid')
 job=dict(path=selector,family=family,scenario=scene,scale=scale,friction=mu,reference=False)
 source=BASE/selector/'complete.json';record=read(source)
 if record['job']!=job or record['protocol_sha256']!=sha(REPORT/'protocol.json'):raise ValueError('Saved declared prediction metadata mismatch')
 from scripts.coupled_contact_v13.surface_pipeline import simulation_settings
 if record['simulation_settings']!=simulation_settings(job,protocol):raise ValueError('Saved prediction settings mismatch')
 evidence[str(source.relative_to(ROOT))]=sha(source)
 return job,evidence

def fresh_output(path):
 path=path.expanduser().absolute()
 if path.exists() or path.is_symlink():raise ValueError('Output must not exist')
 resolved=path.resolve()
 protected=[ROOT/'scripts',ROOT/'configs',ROOT/'reports',ROOT/'papers',BASE,ROOT/'runs/coupled-contact-v13',ROOT/'runs/coupled-friction-v13',ROOT/'runs/third_party']
 if resolved==ROOT or ROOT.is_relative_to(resolved) or any(resolved.is_relative_to(p.resolve()) for p in protected):raise ValueError('Output overlaps source, frozen evidence or active campaign')
 return resolved

def worker(output):
 out=Path(output);request=read(out/'replay-request.json')
 budget=request['deadline_unix']-time.time()
 if budget<=0:raise RuntimeError('Independent replay budget exhausted')
 def expire(signum,frame):
  try:write(out/'replay-timeout.json',dict(reason='Independent child wall-clock deadline',deadline_unix=request['deadline_unix'],observed_unix=time.time()))
  finally:os.killpg(os.getpgrp(),signal.SIGKILL)
 signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,budget)
 freeze,p=verify()
 if request['replay_source_sha256']!=sha(Path(__file__)) or request['method_freeze_sha256']!=sha(REPORT/'method-freeze.json'):raise RuntimeError('Replay request source/freeze mismatch')
 job,evidence=select_job(request['job']['path'],p)
 if job!=request['job'] or evidence!=request['selection_evidence_sha256']:raise RuntimeError('Replay job evidence changed')
 name='scripts.coupled_contact_v13._single_job_replay_surface'
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/surface_pipeline.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 module.BASE=out/'outputs';module.REPORT=REPORT;impl=module.implementation()
 impl.remaining=lambda:max(0.,request['deadline_unix']-time.time())
 # Cache directories are output destinations, not numerical settings; mutation is
 # confined to this dedicated subprocess, never the running campaign process.
 from scripts.coupled_contact_v13 import clipped_surface_contact
 clipped_surface_contact.ROOT=out/'compiler-cache-root'
 if job['family']=='native':
  import warp as wp
  wp.config.kernel_cache_dir=str(out/'warp-cache')
 if impl.remaining()<=0:raise RuntimeError('Independent replay budget exhausted')
 result=module.simulate_job(job)
 if result['simulation_settings']!=request['simulation_settings']:raise RuntimeError('Replay settings differ')
 write(out/'replay-result.json',dict(success=result['success'],job=job,simulation_settings=result['simulation_settings'],status=result,artifact_sha256={str(f.relative_to(out)):sha(f) for f in (out/'outputs'/job['path']).glob('*') if f.is_file()}))
 return 0 if result['success'] else 1

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True,help='Exact declared path, e.g. library/fem/0-0');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args()
 if not 0<args.max_seconds<=43200:raise ValueError('Independent runtime must be >0 and <=12 hours')
 out=fresh_output(args.output);freeze,p=verify();job,evidence=select_job(args.job,p)
 from scripts.coupled_contact_v13.surface_pipeline import simulation_settings
 settings=simulation_settings(job,p);env_record=environment();out.mkdir(parents=True,exist_ok=False);start=time.time()
 request=dict(environment=env_record,scope='One independent replay, not a restarted or completed full study',job=job,simulation_settings=settings,protocol_sha256=sha(REPORT/'protocol.json'),method_freeze_sha256=sha(REPORT/'method-freeze.json'),physics_qualification_sha256=freeze['physics_qualification_sha256'],source_sha256=freeze['source_sha256'],selection_evidence_sha256=evidence,replay_source_sha256=sha(Path(__file__)),max_seconds=args.max_seconds,deadline_unix=start+args.max_seconds,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(out/'replay-request.json',request)
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(out/'cache'))
 code='from scripts.coupled_contact_v13.replay_surface_job import worker; import sys; sys.exit(worker(sys.argv[1]))'
 with (out/'replay.log').open('w') as log:
  process=subprocess.Popen([sys.executable,'-B','-c',code,str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  timeout=False
  try:returncode=process.wait(timeout=max(.001,request['deadline_unix']-time.time()))
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(process.pid,signal.SIGKILL);returncode=process.wait()
 timeout=timeout or (out/'replay-timeout.json').exists()
 write(out/'replay-execution.json',dict(returncode=returncode,timed_out=timeout,elapsed_seconds=time.time()-start,request_sha256=sha(out/'replay-request.json'),replay_result_sha256=sha(out/'replay-result.json') if (out/'replay-result.json').exists() else None,scope=request['scope']))
 return 124 if timeout else returncode
if __name__=='__main__':raise SystemExit(main())
