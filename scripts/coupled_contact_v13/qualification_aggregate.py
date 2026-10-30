"""Derive per-scenario qualification from hashed development evidence.

This aggregator cannot promote unresolved drop evidence or choose a study cohort.
"""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.campaign import REPORT,write
MATERIALS={(6000.,.35),(3000.,.1)}
SETTINGS={'squeeze':(8,10),'slide':(8,10),'curved':(8,10),'moving_sphere':(12,10)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def source_ok(d):
 h=d['plan']['source_sha256']
 return bool(d.get('source_unchanged',False)) and all((ROOT/p).exists() and sha(ROOT/p)==v for p,v in h.items())
def gate(p,d,indices,kind):
 checks=[d['checks'][i] for i in indices];covered={(c.get('young'),c.get('friction')) for c in checks}
 if kind=='endpoints':
  covered={(c['spec'][1],c['spec'][2]) for c in checks};coverage=covered=={(E,mu) for E in [2000.,18000.] for mu in [0.,.8]} and len(checks)==4
 else:coverage=covered==MATERIALS and len(checks)==2
 valid=source_ok(d);passed=coverage and valid and all(c['passed'] for c in checks)
 return dict(report=str(p.relative_to(ROOT)),report_sha256=sha(p),selected_check_indices=indices,expected_coverage_pass=coverage,source_hashes_verified=valid,passed=passed)
def spatial(name,pair):
 p=REPORT/('consistent-higher-refinement.json' if name=='curved' else 'consistent-other-spatial.json');d=load(p)
 prior=load(REPORT/'consistent-mass-diagnostic.json')
 settings_ok=d['plan']['dt']==2.5e-5 and sha(REPORT/'consistent-mass-diagnostic.json')==d['plan']['prior_report_sha256'] and prior['plan']['friction_level']==1 and prior['plan']['sphere_spacing_m']==.0015
 ix=[i for i,c in enumerate(d['checks']) if c['scenario']==name and c['cells']==sorted(pair)]
 result=gate(p,d,ix,'spatial');result['settings_verified']=settings_ok;result['passed'] &= settings_ok
 return result
def reference(name,n,mode):
 matches=[]
 for p in sorted(REPORT.glob(f'consistent-reference-{mode}-*.json')):
  d=load(p)
  if 'checks' not in d or d['plan']['reference_map'].get(name)!=n or d['plan']['dt']!=2.5e-5:continue
  ix=[i for i,c in enumerate(d['checks']) if (c['spec'][0] if mode=='endpoints' else c['scenario'])==name]
  base_specs=[s for s in d['plan']['specs'] if s[0]==name and s[3]==n and s[4]==2.5e-5]
  settings_ok=bool(base_specs) and all(s[5]==.0015 and s[6]==1 for s in base_specs if mode!='contact')
  if mode=='contact':settings_ok=bool(base_specs) and all((s[5],s[6]) in [(.0015,1),(.0015,2),(.00075,1)] for s in base_specs) and any(s[5]==.0015 and s[6]==1 for s in base_specs)
  result=gate(p,d,ix,mode);result['settings_verified']=settings_ok;result['passed'] &= settings_ok;matches.append(result)
 if len(matches)>1:raise RuntimeError(f'Ambiguous reference evidence: {name}/{n}/{mode}')
 return matches[0] if matches else dict(passed=False,pending=True,reason=f'Missing {name} cells{n} {mode} evidence')
def main():
 predictor_reports=['consistent-predictor-halfstep.json','predictor-selected-contact.json','consistent-predictor-endpoint-screen.json']
 predictor=[]
 for name in predictor_reports:
  p=REPORT/name
  if not p.exists():predictor.append(dict(pending_report=str(p.relative_to(ROOT)),passed=False));continue
  d=load(p);coverage=True
  if name=='predictor-selected-contact.json':
   expected={(n,E,mu) for n in ['squeeze','edge_drop','curved','moving_sphere'] for E,mu in MATERIALS}
   coverage={(c['scenario'],c['young'],c['friction']) for c in d['checks']}==expected and len(d['checks'])==8 and d['plan']['cells']==4 and d['plan']['dt']==1.25e-5 and d['plan']['baseline_spacing']==.0015 and d['plan']['baseline_friction_level']==1
  predictor.append(dict(report=str(p.relative_to(ROOT)),report_sha256=sha(p),expected_coverage_pass=coverage,source_hashes_verified=source_ok(d),passed=coverage and source_ok(d) and d['all_gates_pass']))
 scenarios={}
 for name,(n,c) in SETTINGS.items():
  gates={'spatial':spatial(name,[n,c]),'primary_time':reference(name,n,'time'),'companion_time':reference(name,c,'time'),'contact':reference(name,n,'contact'),'endpoints':reference(name,n,'endpoints')}
  scenarios[name]=dict(primary_cells=n,companion_cells=c,dt=2.5e-5,half_dt=1.25e-5,predictor_cells=4,predictor_dt=1.25e-5,spatial_pair=sorted([n,c]),surface_spacing_m=.0015,friction_level=1,gates=gates,source_hashes_verified=all(g.get('source_hashes_verified',False) for g in gates.values()),qualified=all(g['passed'] for g in gates.values()) and all(g['passed'] for g in predictor),pending=[k for k,g in gates.items() if not g['passed']]+(['predictor_evidence'] if not all(g['passed'] for g in predictor) else []))
 # Drop adoption is deliberately outside this aggregator until all corresponding
 # gates exist. A narrowed study must declare its cohort independently.
 scenarios['sharp_edge_drop']=dict(qualified=False,pending=['No accepted spatial pair'],evidence_report=str((REPORT/'consistent-higher-refinement.json').relative_to(ROOT)),evidence_report_sha256=sha(REPORT/'consistent-higher-refinement.json'))
 scenarios['rounded_support_drop']=dict(qualified=False,pending=['Initial spatial pair failed; bounded fallback/time diagnosis pending; further qualification required'])
 evidence_paths={g['report'] for g in predictor if 'report' in g}
 for scene in scenarios.values():
  evidence_paths.update(g['report'] for g in scene.get('gates',{}).values() if 'report' in g)
 union={}
 for evidence in sorted(evidence_paths):
  for path,digest in load(ROOT/evidence)['plan']['source_sha256'].items():
   if path in union and union[path]!=digest:raise RuntimeError(f'Conflicting historical source hashes for {path}')
   union[path]=digest
 if not all((ROOT/path).exists() and sha(ROOT/path)==digest for path,digest in union.items()):raise RuntimeError('Qualification source union differs from current files')
 self_path=Path(__file__);union[str(self_path.relative_to(ROOT))]=sha(self_path)
 result=dict(nesting_amendment=dict(path=str((REPORT/'consistent-reference-nesting-amendment.json').relative_to(ROOT)),sha256=sha(REPORT/'consistent-reference-nesting-amendment.json')),source_sha256=union,schema_version=1,status='in_progress',updated_utc=datetime.now(timezone.utc).isoformat(),all_gates_pass=all(s['qualified'] for s in scenarios.values()),scenarios=scenarios,predictor_evidence=predictor,aggregator_source_sha256={str(self_path.relative_to(ROOT)):sha(self_path)},scope='Qualification evidence only; no automatic cohort selection, no relaxed gates, no promotion of unresolved stress cases',spatial_claim='Two-resolution consistency, not continuum convergence')
 write(ROOT/'reports/coupled-surface-v13/physics-development-qualification.json',result)
 print(json.dumps({name:s['qualified'] for name,s in scenarios.items()}))
if __name__=='__main__':main()
