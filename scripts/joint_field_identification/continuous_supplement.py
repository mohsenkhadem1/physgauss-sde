"""Posthoc higher-particle numerical check; original cohorts and flags retained."""
import datetime,hashlib,itertools,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/joint-field-surface-v04/continuous';OUT=REPORT/'posthoc4096';MAN=OUT/'manifest.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def name(j):return f"{j['shape']}-{j['population']}-case{j['case']:03d}-{j['stage']}-n{j['particles']}-s{j['seed']}"
def analyze():
 docs=[read(p) for p in sorted(list(REPORT.glob('block-matched-case001-fixed-*.json'))+list(OUT.glob('block-matched-case001-fixed-*.json')))];rows=[];pairs=[]
 for d in docs:
  a=np.load(ROOT/d['prediction_archive']);rows.append(dict(job=name(d['job']),parameter_mean=d['parameter_mean'],parameter_correlation=np.corrcoef(a['theta'].T).tolist(),log_evidence=d['log_evidence'],final_acceptance=d['stages'][-1]['acceptance_rate'],unique_particles=d['unique_particles'],particles=d['particles'],invalid_evaluations=d['invalid_evaluation_count'],weight_ess=d['weight_ess'],report_sha256=sha((OUT if d['job']['particles']==4096 else REPORT)/(name(d['job'])+'.json'))))
 for d,e in itertools.combinations(docs,2):
  a=np.load(ROOT/d['prediction_archive']);b=np.load(ROOT/e['prediction_archive']);diff=np.abs(a['cdf']-b['cdf']);rms=float(np.sqrt(np.mean((a['target_mean']-b['target_mean'])**2))*1000);pairs.append(dict(first=name(d['job']),second=name(e['job']),cdf_by_parameter=np.max(diff,axis=1).tolist(),max_cdf=float(np.max(diff)),argmax_parameter=int(np.unravel_index(np.argmax(diff),diff.shape)[0]),prediction_mean_rms_mm=rms,flag_cdf=bool(np.max(diff)>.1),flag_prediction=rms>.01))
 save(OUT/'diagnosis.json',dict(updated_utc=now(),manifest_sha256=sha(MAN),parameter_names=['logE','material_contrast','support_center_x','log_support_stiffness','load_center_x'],records=rows,comparisons=pairs,interpretation=['Original1024/2048 flags retained. Targeted posthoc diagnostic, not independent confirmation or replacement.','Strong support-load/stiffness correlations and ~12% final Metropolis acceptance indicate limited ridge exploration; no distinct missed mode is established by these summaries.','Equal resampled weights do not establish convergence.','All4096 runs use unchanged four Metropolis moves and same observations/model; only particle count and independent seeds differ.','CDF threshold.1 and predictionmeanRMS.01mm remain descriptive, not exclusion criteria.']))
def main():
 m=read(MAN);state=dict(started_utc=now(),manifest_sha256=sha(MAN),pid=os.getpid(),status='waiting_for_original_campaigns',failures={});deadline=datetime.datetime.fromisoformat(m['deadline_utc'].replace('Z','+00:00')).timestamp();children=[];logs=[]
 def update():state['updated_utc']=now();save(OUT/'status.json',state)
 analyze()
 while time.time()<deadline:
  ready=all(read(ROOT/'reports'/r/'continuous-summary.json')['complete'] for r in ['joint-field-identification-v01','joint-field-surface-v04']) and read(REPORT/'dynamic-dispatch.json')['status']=='complete'
  if ready:break
  update();time.sleep(30)
 if time.time()>=deadline:state['status']='deadline_before_launch';update();return
 for f,h in m['source_sha256'].items():assert sha(ROOT/f)==h,f
 for i,j in enumerate(m['allowed_jobs']):
  log=(OUT/(name(j)+'.log')).open('a');logs.append(log);cmd=[sys.executable,'-m','scripts.joint_field_identification.surface_continuous_posthoc','--shape',j['shape'],'--population',j['population'],'--case',str(j['case']),'--stage',j['stage'],'--particles',str(j['particles']),'--seed',str(j['seed']),'--cells','8','4','4','--device',f'cuda:{i}','--batch','16','--moves','4','--manifest',str(MAN),'--resume'];p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});children.append((j,p));state.setdefault('workers',[]).append(dict(job=name(j),pid=p.pid,device=f'cuda:{i}'))
 state['status']='running';update()
 while any(p.poll() is None for _,p in children) and time.time()<deadline:update();time.sleep(15)
 for j,p in children:
  if p.poll() is None:p.terminate();p.wait(timeout=30)
  if p.returncode:state['failures'][name(j)]=p.returncode
 for log in logs:log.close()
 analyze();state['status']='complete' if not state['failures'] else 'finished_with_failures';update()
if __name__=='__main__':main()
