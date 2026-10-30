"""Separately frozen friction-grid ablation; original study remains read-only.

Same physical solver, observations, loss and cohorts. Exact old library/oracle
artifacts are explicitly linked with provenance. No reference generation exists.
All stages require independent pre-freeze approval; six CPU workers maximum.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util,concurrent.futures,multiprocessing,argparse,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13 import surface_pipeline as parent_surface
from scripts.coupled_contact_v13.campaign import write,remaining
BASE=ROOT/'runs/coupled-friction-v13';REPORT=ROOT/'reports/coupled-friction-v13'
PARENT_BASE=ROOT/'runs/coupled-surface-v13';PARENT_REPORT=ROOT/'reports/coupled-surface-v13'
_PRIVATE=None

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def protocol():return json.loads((REPORT/'protocol.json').read_text())
def original():return parent_surface.implementation()
def safe_path(base,relative):
 p=Path(relative)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Relative repository job path required')
 return base/p

def check_freeze():
 original().check_freeze()
 p=protocol();parent=original().protocol();ext=p['fine_grid_extension']
 for name,key in [('protocol.json','parent_protocol_sha256'),('method-freeze.json','parent_method_freeze_sha256'),('library-coverage-audit.json','parent_library_audit_sha256')]:
  if digest(PARENT_REPORT/name)!=ext[key]:raise RuntimeError('Parent evidence changed: '+name)
 allowed={'friction_grid','scope','fine_grid_extension'}
 if set(p)-set(parent)!={'fine_grid_extension'} or any(p[k]!=v for k,v in parent.items() if k not in allowed):raise RuntimeError('Fine study changes more than declared friction grid/scope')
 if p['friction_grid']!=[i/20 for i in range(17)] or not set(parent['friction_grid'])<=set(p['friction_grid']):raise RuntimeError('Exact nested uniform grid required')
 if len(p['fem_young_grid'])!=7 or len(p['native_scale_grid'])!=7:raise RuntimeError('Declared7-scale grids required')
 freeze=json.loads((REPORT/'method-freeze.json').read_text());approval=json.loads((REPORT/'prefreeze-audit.json').read_text())
 if freeze['protocol_sha256']!=digest(REPORT/'protocol.json'):raise RuntimeError('Fine protocol changed')
 for key in ('parent_protocol_sha256','parent_method_freeze_sha256'):
  if freeze.get(key)!=ext[key]:raise RuntimeError('Fine freeze parent binding differs')
 if 'scripts/coupled_contact_v13/fine_grid_pipeline.py' not in freeze['source_sha256']:raise RuntimeError('Fine driver absent from freeze')
 for path,sha in freeze['source_sha256'].items():
  if digest(safe_path(ROOT,path))!=sha:raise RuntimeError('Fine frozen source changed: '+path)
 if approval.get('passed') is not True or approval.get('errors') or approval.get('pending'):raise RuntimeError('Independent completed prefreeze approval required')
 if approval.get('protocol_sha256')!=digest(REPORT/'protocol.json') or approval.get('method_freeze_sha256')!=digest(REPORT/'method-freeze.json'):raise RuntimeError('Fine approval is stale')
 if remaining()<=0:raise RuntimeError('Original campaign deadline reached')

def private_surface():
 global _PRIVATE
 if _PRIVATE is None:
  original()  # Retain original module object before private pipeline import.
  name='scripts.coupled_contact_v13._fine_grid_private_surface'
  spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/surface_pipeline.py')
  module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  module.BASE=BASE;module.REPORT=REPORT
  impl=module.implementation();impl.protocol=protocol;impl.check_freeze=check_freeze
  impl.execute=execute;impl.simulate_job=simulate_job;impl.prepare_object=prepare_object
  impl.reference_quality=reference_quality  # Read-only completed parent evaluation, no parent writes.
  _PRIVATE=module
 return _PRIVATE

def implementation():return private_surface().implementation()

def reference_quality(object_id,scenario):
 obj=next(o for o in protocol()['objects'] if o['id']==object_id)
 path=PARENT_REPORT/f"{obj['split']}-results.json"
 report=json.loads(path.read_text());rows=[r for r in report['cases'] if r['object']==object_id]
 if len(rows)!=len(protocol()['forecast_scenarios']):raise RuntimeError('Complete parent object accounting required')
 if scenario=='squeeze':
  qualities=[dict(qualified=r['calibration_reference_qualified'],reasons=r['calibration_qualification_reasons']) for r in rows]
  if any(q!=qualities[0] for q in qualities):raise RuntimeError('Parent calibration quality inconsistent')
  result=qualities[0]
 else:
  matches=[r for r in rows if r['scenario']==scenario]
  if len(matches)!=1:raise RuntimeError('Unique parent forecast row required')
  row=matches[0];result=dict(qualified=row['reference_qualified'],reasons=row['reference_qualification_reasons'],rms_m=row.get('reference_refinement_rms_m'))
 result['parent_evaluation']=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
 return result

def link_exact(source,destination):
 source=source.resolve();destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists() or destination.is_symlink():
  if not destination.is_symlink() or destination.resolve()!=source:raise RuntimeError('Unexpected existing reuse destination: '+str(destination))
 else:destination.symlink_to(source,target_is_directory=source.is_dir())

def artifact_hashes(path):
 return {name:digest(path/name) for name in ['diagnostics.json','trajectory.npz'] if (path/name).exists()}

def validate_reuse(record,destination):
 reuse=record.get('reuse')
 if not reuse:return
 ext=protocol()['fine_grid_extension'];source=safe_path(PARENT_BASE,reuse['source_job']['path'])
 if reuse['parent_protocol_sha256']!=ext['parent_protocol_sha256'] or reuse['parent_method_freeze_sha256']!=ext['parent_method_freeze_sha256']:raise RuntimeError('Reuse parent mismatch')
 if digest(source/'complete.json')!=reuse['source_complete_sha256']:raise RuntimeError('Reused completion changed')
 source_status=json.loads((source/'complete.json').read_text())
 if source_status['job']!=reuse['source_job'] or source_status['protocol_sha256']!=ext['parent_protocol_sha256']:raise RuntimeError('Reused source job/protocol differs')
 for key in ['family','scenario','scale','friction','reference']:
  if source_status['job'][key]!=record['job'][key]:raise RuntimeError('Reuse parameters differ')
 if source_status['simulation_settings']!=record['simulation_settings'] or source_status['success']!=record['success']:raise RuntimeError('Reuse physics/status differs')
 for name,sha in reuse['source_file_sha256'].items():
  if name not in ['complete.json','diagnostics.json','trajectory.npz'] or digest(source/name)!=sha:raise RuntimeError('Reused artifact changed')
  if name!='complete.json' and (not (destination/name).is_symlink() or (destination/name).resolve()!=(source/name).resolve() or digest(destination/name)!=sha):raise RuntimeError('Reuse link is not exact')

def validate_cached(job):
 path=safe_path(BASE,job['path']);record=json.loads((path/'complete.json').read_text());settings=parent_surface.simulation_settings(job,protocol())
 if record['job']!=job or record['simulation_settings']!=settings or record['protocol_sha256']!=digest(REPORT/'protocol.json'):raise RuntimeError('Fine cache mismatch')
 validate_reuse(record,path)
 expected={'diagnostics.json','trajectory.npz'} if record['success'] else set()
 if set(record.get('artifact_sha256',{}))!=expected:raise RuntimeError('Fine artifact inventory mismatch')
 for name,sha in record['artifact_sha256'].items():
  if digest(path/name)!=sha:raise RuntimeError('Fine output changed')
 return record

def reuse_source(job):
 if job['family']=='oracle':return dict(job)
 if not job['path'].startswith('library/'):return None
 parent=original().protocol()
 if job['friction'] not in parent['friction_grid']:return None
 scales=parent['fem_young_grid'] if job['family']=='fem' else (parent['native_scale_grid'] if job['family']=='native' else [6000.])
 i=scales.index(job['scale']);j=parent['friction_grid'].index(job['friction'])
 return dict(job,path=f"library/{job['family']}/{i}-{j}")

def reuse_job(job,source_job):
 source=safe_path(PARENT_BASE,source_job['path']);destination=safe_path(BASE,job['path'])
 record=json.loads((source/'complete.json').read_text());ext=protocol()['fine_grid_extension']
 if record['job']!=source_job or record['protocol_sha256']!=ext['parent_protocol_sha256'] or record['simulation_settings']!=parent_surface.simulation_settings(job,protocol()):raise RuntimeError('Source reuse job/settings mismatch')
 files={'complete.json':digest(source/'complete.json'),**artifact_hashes(source)}
 if job['path'].startswith('library/'):
  audit=json.loads((PARENT_REPORT/'library-coverage-audit.json').read_text())
  if audit.get('passed') is not True or audit.get('errors'):raise RuntimeError('Parent library audit required')
  candidate=[r for r in audit['candidates'] if r['expected_job']==source_job]
  if len(candidate)!=1:raise RuntimeError('Unique audited parent candidate required')
  if {str((source/name).relative_to(ROOT)):sha for name,sha in files.items()}!=candidate[0]['file_sha256']:raise RuntimeError('Parent candidate files differ from audit')
 destination.mkdir(parents=True,exist_ok=True)
 for name in files:
  if name!='complete.json':link_exact(source/name,destination/name)
 copied=copy.deepcopy(record);copied.update(job=job,protocol_sha256=digest(REPORT/'protocol.json'),artifact_sha256={n:h for n,h in files.items() if n!='complete.json'},
  reuse=dict(source_job=source_job,source_complete_sha256=files['complete.json'],source_file_sha256=files,parent_protocol_sha256=ext['parent_protocol_sha256'],parent_method_freeze_sha256=ext['parent_method_freeze_sha256']))
 write(destination/'complete.json',copied);return validate_cached(job)

def simulate_job(job):
 check_freeze()
 if job['reference']:raise RuntimeError('Fine study cannot generate or overwrite references')
 path=safe_path(BASE,job['path'])
 if (path/'complete.json').exists():return validate_cached(job)
 source=reuse_source(job)
 if source is not None:return reuse_job(job,source)
 record=private_surface().simulate_job(job)
 record['artifact_sha256']=artifact_hashes(path) if record['success'] else {}
 write(path/'complete.json',record);return validate_cached(job)

def execute(jobs,workers):
 if not 1<=workers<=6:raise ValueError('At most6 CPU workers authorized')
 jobs=list(jobs)
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  rows=[]
  for row in pool.map(simulate_job,jobs):
   rows.append(row);print(json.dumps(dict(job=row['job'],success=row['success'],reused='reuse' in row)),flush=True)
 return rows

def library_jobs():return implementation().library_jobs()

def verify_library():
 jobs=library_jobs()
 if len(jobs)!=255:raise RuntimeError('Expected255 library jobs')
 for job in jobs:validate_cached(job)

def link_references(split):
 for obj in protocol()['objects']:
  if obj['split']!=split:continue
  for scene in [protocol()['calibration_scenario']]+protocol()['forecast_scenarios']:
   source=PARENT_BASE/'reference'/obj['id']/scene
   if not (source/'complete.json').exists():raise RuntimeError('Original reference job is pending')
   link_exact(source,BASE/'reference'/obj['id']/scene)

def prepare_object(obj):
 m=implementation();source=PARENT_BASE/'inference'/obj['id'];calibration=PARENT_BASE/'reference'/obj['id']/'squeeze'
 status=json.loads((calibration/'complete.json').read_text())
 if not status['success']:return None,None,None
 observation=source/'observations.npz';observation_sha=digest(observation)
 with np.load(observation,allow_pickle=False) as d:
  rest=d['rest'].copy();observed=d['observed'].copy();mask=d['mask'].copy();tools=d['tools'].copy();times=d['times'].copy();active=d['active'].copy()
 coarse=m.build('squeeze',cells=tuple(protocol()['coarse_cells']))
 if not np.array_equal(rest,coarse.rest):raise RuntimeError('Original observation rest geometry differs')
 selected,empirical=m.calibrate(rest,observed,mask,tools,times,active,m.load_libraries())
 if digest(observation)!=observation_sha:raise RuntimeError('Observation changed during fitting')
 destination=BASE/'inference'/obj['id'];link_exact(observation,destination/'observations.npz')
 write(destination/'selection.json',selected)
 write(destination/'observation-reuse.json',dict(source=str(observation.relative_to(ROOT)),sha256=observation_sha,parent_protocol_sha256=protocol()['fine_grid_extension']['parent_protocol_sha256']))
 return selected,empirical,coarse

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['library','select','evaluate']);parser.add_argument('--split',choices=['development','reserved'],default='development');parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
 check_freeze()
 if not 1<=args.workers<=6:raise ValueError('At most6 workers')
 if args.stage=='library':
  write(REPORT/'library-status.json',execute(library_jobs(),args.workers));return
 verify_library();link_references(args.split)
 if args.stage=='select':
  for obj in protocol()['objects']:
   if obj['split']==args.split:prepare_object(obj)
 else:
  if args.split=='reserved' and not (REPORT/'development-results.json').exists():raise RuntimeError('Development forecast accounting required before reserved evaluation')
  parent_result=PARENT_REPORT/f'{args.split}-results.json'
  if not parent_result.exists():raise RuntimeError('Parent coarse accounting pending')
  parent_sha=digest(parent_result)
  result=implementation().evaluate(args.split,args.workers)
  if digest(parent_result)!=parent_sha:raise RuntimeError('Parent accounting changed during evaluation')
  result['grid_ablation_scope']='Separate fixed uniform friction refinement; no matched fine-grid vertex claim; parent coarse study unchanged.'
  parent_result=PARENT_REPORT/f'{args.split}-results.json'
  if not parent_result.exists():raise RuntimeError('Parent coarse accounting pending')
  result['parent_coarse_results']=dict(path=str(parent_result.relative_to(ROOT)),sha256=digest(parent_result))
  write(REPORT/f'{args.split}-results.json',result)

if __name__=='__main__':main()
