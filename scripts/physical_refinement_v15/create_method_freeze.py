"""One-shot prospective source/artifact freeze; never generates reserved data."""
import ast,hashlib,json,sys,platform,datetime,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/physical-refinement-v15'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def resolve(module):
 out=[]
 for base in (ROOT,ROOT/'src'):
  p=base.joinpath(*module.split('.'))
  for q in (p.with_suffix('.py'),p/'__init__.py'):
   if q.is_file():out.append(q)
 return out

def main():
 freeze=REPORT/'method-freeze.json'
 if freeze.exists():raise ValueError('Preserve existing method freeze')
 if (ROOT/'runs/physical-refinement-v15/reserved').exists():raise ValueError('Reserved tree already exists')
 protocol=REPORT/'final-protocol.json';p=read(protocol)
 assert len(p['reserved_worlds'])==12 and p['gradient_count']==17 and p['scale_count']==97
 assert p['noise_repeats']==4 and p['primary_budget']==2 and p['tracks']==['V','VF']
 transcript=REPORT/'prefreeze-final-tests.txt'
 assert 'passed' in transcript.read_text() and 'failed' not in transcript.read_text().lower()
 prior=read(REPORT/'prior-preservation.json')['sha256']
 for name,h in prior.items():assert sha(ROOT/name)==h,name
 seeds=list((ROOT/'scripts/physical_refinement_v15').glob('*.py'))+list((ROOT/'tests').glob('test_physical_refinement*.py'))
 pending=list(seeds);sources={};imports={}
 while pending:
  file=pending.pop().resolve();name=str(file.relative_to(ROOT))
  if name in sources:continue
  sources[name]=sha(file);tree=ast.parse(file.read_text());found=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):mods=[a.name for a in node.names]
   elif isinstance(node,ast.ImportFrom):
    if node.level:
     parent=file.parent
     for _ in range(node.level-1):parent=parent.parent
     rel=parent.relative_to(ROOT);base='.'.join(rel.parts)
     module=base+('.'+node.module if node.module else '')
    else:module=node.module or ''
    mods=[module]+[module+'.'+a.name for a in node.names]
   else:continue
   for mod in mods:
    for path in resolve(mod):
     pending.append(path);found.append(str(path.relative_to(ROOT)))
  imports[name]=sorted(set(found))
 # Imported mechanics modules include local statically referenced experiment code.
 required=['scripts/phystwin/active_support_reference.py','experiments/campaign_static_p2_block.py','experiments/campaign_static_uniaxial_bar.py']
 for name in required:assert name in sources,('Missing required dependency closure',name)
 artifacts={}
 def add(path,digest=None):
  path=Path(path);path=path if path.is_absolute() else ROOT/path
  name=str(path.resolve().relative_to(ROOT));h=sha(path)
  if digest is not None:assert h==digest,name
  if name in artifacts:assert artifacts[name]==h
  artifacts[name]=h
 for report in ['bank-semantic-audit-g17.json','independent-mechanics-gate-audit.json']:
  d=read(REPORT/report);assert d['passed'] is True
  add(REPORT/report)
  for path,h in d['evidence_sha256'].items():add(path,h)
 for path in (ROOT/'runs/physical-refinement-v15/response-predictions').rglob('*'):
  if path.is_file() and path.suffix in ('.json','.npz'):add(path)
 for name,h in prior.items():add(name,h)
 for name,h in p['development_evidence_sha256'].items():add(name,h)
 # Preserve pre-test methodological decisions, failures and qualification evidence.
 for path in REPORT.iterdir():
  if path.is_file() and path.suffix in ('.json','.txt') and path.name not in ('method-freeze.json','prefreeze-independent-audit.json','source-inventory.json'):add(path)
 inventory=REPORT/'source-inventory.json'
 write(inventory,dict(scope='All v15 runtime modules/tests plus recursively resolved local AST import closure; external libraries recorded separately',source_sha256=dict(sorted(sources.items())),local_import_edges=imports))
 add(inventory)
 for name,h in sources.items():assert sha(ROOT/name)==h,name
 for name,h in artifacts.items():assert sha(ROOT/name)==h,name
 packages={}
 for name in ('numpy','scipy','torch','pyamg','scikit-fem','pytest'):
  try:packages[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:packages[name]=None
 result=dict(role='prospective_exploratory_method_freeze',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(protocol),source_sha256=dict(sorted(sources.items())),artifact_sha256=dict(sorted(artifacts.items())),source_inventory_sha256=sha(inventory),tests_sha256=sha(transcript),prior_preserved_files=len(prior),python=sys.version,python_executable=sys.executable,platform=platform.platform(),package_versions=packages,scope='Finite g17/scale97 physical belief; failed continuous posterior and renderer convergence retained. No reserved data generated before freeze.')
 write(freeze,result)
 audit=dict(passed=True,errors=[],created_utc=result['created_utc'],protocol_sha256=sha(protocol),method_freeze_sha256=sha(freeze),source_count=len(sources),artifact_count=len(artifacts),prior_preserved_files=len(prior),tests=transcript.read_text().strip(),numerical_gate_audit_sha256=sha(REPORT/'independent-mechanics-gate-audit.json'),reserved_tree_absent_at_freeze=True,scope='Independent source/data-separation review plus exact immutable inventories and232 saved-array mechanics checks; exploratory scope explicitly retained. No calibrated continuous-posterior or physical-camera qualification claim.')
 write(REPORT/'prefreeze-independent-audit.json',audit)
 print(json.dumps({k:audit[k] for k in ('passed','protocol_sha256','method_freeze_sha256','source_count','artifact_count','prior_preserved_files')}))
if __name__=='__main__':main()
