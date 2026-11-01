"""Separate globally development-selected Gaussian family; fixed all-case method.
No per-case outcome/validation switch. Scoring is separately gated on all eight
committed forecasts and an explicit root authorization artifact.
"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import ROOT,BASE,REPORT,read,write,sha,remaining
from scripts.benchmark_improvement_v14.provenance import snapshot_sources
from scripts.benchmark_improvement_v14.phystwin_production import NAMES,declared,strict
from scripts.benchmark_improvement_v14.phystwin_carrier_gaussian import predict,SETTINGS
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.experiment import finite_metadata
from scripts.external_benchmark.data import interpolate_matrix
from scripts.contact_benchmark_v12.feasibility import inherited_elements
from scripts.contact_benchmark_v12.feasible_experiment import topology_audit
FREEZE=REPORT/'phystwin-carrier-gaussian-freeze.json'
PARAMETERS=dict(SETTINGS,representation='Direct released support-space field; nodal query saved separately for topology',policy='Always-on same family for all eight; input/numerical/nonfinite failure retains original support-space object_frame',scope='Separate globally development-selected exploratory carrier family; development manual error worse, SHORT and shape better; no independent confirmation or physical-admissibility claim')

def folder(case):return BASE/'carrier-gaussian-evaluation'/case['cohort']/case['name']

def anchor_folder(case):
 return (ROOT/'runs/external-benchmark-v11/fits/primary' if case['cohort']=='regression' else ROOT/'runs/contact-benchmark-v12/base/full')/case['name']

def inventory(case):
 result=snapshot_sources(case);base=anchor_folder(case)
 for file in ('prediction.npz','prediction.json','provenance.json','fit.json','best.npz','progress.json'):result[str(base/file)]=sha(base/file)
 if (base/'complete.json').exists():result[str(base/'complete.json')]=sha(base/'complete.json')
 return result

def prepare():
 remaining();cases=[declared(name) for name in NAMES]
 # Inherit existing verified implementation dependency coverage, then add the
 # exact Gaussian predictor and isolated production/scoring wrapper.
 original=read(REPORT/'phystwin-freeze.json');sources=dict(original['source_sha256'])
 for path,digest in sources.items():
  if sha(ROOT/path)!=digest:raise ValueError('Parent frozen source changed: '+path)
 for path in (Path(__file__),ROOT/'scripts/benchmark_improvement_v14/phystwin_carrier_gaussian.py',ROOT/'scripts/benchmark_improvement_v14/phystwin_carrier_gaussian.py'):
  sources[str(path.relative_to(ROOT))]=sha(path)
 proposal=dict(schema_version=1,protocol_sha256=sha(REPORT/'protocol.json'),parent_width_freeze_sha256=sha(REPORT/'phystwin-freeze.json'),parameters=PARAMETERS,cases=cases,source_sha256=sources,input_sha256={case['name']:inventory(case) for case in cases},development_evidence_sha256={str(path.relative_to(ROOT)):sha(path) for path in (REPORT/'phystwin-carrier-gaussian-plan.json',REPORT/'phystwin-carrier-gaussian-development.json')})
 write(REPORT/'phystwin-carrier-gaussian-freeze-proposal.json',proposal);return proposal

def guard():
 remaining();frozen=read(FREEZE)
 if frozen['protocol_sha256']!=sha(REPORT/'protocol.json') or frozen['parameters']!=PARAMETERS:raise ValueError('Carrier Gaussian protocol/settings changed')
 if frozen['parent_width_freeze_sha256']!=sha(REPORT/'phystwin-freeze.json'):raise ValueError('Parent width freeze changed')
 cases=[declared(name) for name in NAMES]
 if frozen['cases']!=cases or set(frozen['input_sha256'])!=set(NAMES):raise ValueError('Gaussian eight-case cohort changed')
 for path,digest in frozen['source_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Carrier Gaussian frozen source changed: '+path)
 for path,digest in frozen['development_evidence_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Gaussian development evidence changed')
 required={'scripts/benchmark_improvement_v14/phystwin_carrier_gaussian.py','scripts/benchmark_improvement_v14/phystwin_carrier_gaussian_production.py','scripts/contact_benchmark_v12/object_frame.py','scripts/external_benchmark/data.py'}
 if not required.issubset(frozen['source_sha256']):raise ValueError('Missing Gaussian source freeze coverage')
 return frozen,sha(FREEZE)

def validate(case):
 frozen,digest=guard();out=folder(case);record=read(out/'complete.json')
 if record['case']!=case or record['freeze_sha256']!=digest or record['inputs']!=frozen['input_sha256'][case['name']]:raise ValueError('Carrier Gaussian forecast identity/contract changed')
 if inventory(case)!=record['inputs']:raise ValueError('Carrier Gaussian forecast inputs changed')
 for key in ('predictions.npz','nodal.npz','metadata.json'):
  if sha(out/key)!=record['artifact_sha256'][key]:raise ValueError('Carrier Gaussian forecast artifact changed')
 return record

def forecast(case):
 case=declared(case['name']);frozen,digest=guard();out=folder(case)
 if (out/'complete.json').exists():validate(case);return True
 if out.exists():raise ValueError('Do not silently overwrite partial Carrier Gaussian forecast')
 inputs=inventory(case)
 if inputs!=frozen['input_sha256'][case['name']]:raise ValueError('Carrier Gaussian frozen inputs changed')
 original_path=ROOT/'runs/contact-benchmark-v12/evaluation'/case['cohort']/case['name']/'predictions.npz'
 with np.load(original_path) as saved:
  original=saved['object_frame'].copy();inherited=saved['v11_fem'].copy()
 data=load_payload(ROOT/'runs/prefix-ridge-switch-v13/inputs'/case['name']/'full.npz')
 with np.load(anchor_folder(case)/'prediction.npz') as saved:anchor=saved['predicted'].copy()
 if not np.array_equal(anchor,inherited,equal_nan=True):raise ValueError('Original full FEM source differs from pinned original archive')
 support=np.asarray(data['support_geometry']);A=interpolate_matrix(data['nodes'],support)
 original_support=support[None]+np.einsum('pn,tnc->tpc',A,original-data['nodes'][None])
 inherited_support=support[None]+np.einsum('pn,tnc->tpc',A,anchor-data['nodes'][None])
 fallback=False;reason=None
 try:
  strict(data,case['name'])
  if anchor.shape!=original.shape or not np.isfinite(anchor[0]).all():raise ValueError('Inherited initial FEM geometry is invalid')
  proposed,metadata=predict(data,anchor[0],locations=support)
  nodal,nodal_meta=predict(data,anchor[0],locations=data['nodes'])
  if proposed.shape!=original_support.shape or not np.isfinite(proposed).all() or not np.isfinite(nodal).all():raise ValueError('Carrier prediction is nonfinite or wrong shape')
  if not np.allclose(proposed[0],original_support[0],rtol=0,atol=1e-12):raise ValueError('Inherited initial correspondence changed')
  from scipy.spatial import cKDTree
  distances,indices=cKDTree(support).query(data['nodes'])
  if distances.max()!=0:raise ValueError('Mechanical nodes absent from initial support')
  shared_error=float(np.max(np.abs(proposed[:,indices]-nodal)))
  if shared_error>1e-12:raise ValueError('Direct-support/nodal field inconsistency')
  metadata['shared_support_node_max_abs_error_m']=shared_error
 except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
  fallback=True;reason=str(error);proposed=original_support.copy();nodal=original.copy();metadata=dict(failed_candidate=True,reason=reason)
 predictions=dict(object_frame=original_support,carrier_gaussian_v14=proposed,inherited_fem=inherited_support)
 nodal_predictions=dict(object_frame=original,carrier_gaussian_v14=nodal,inherited_fem=anchor)
 elements=inherited_elements(data)
 metadata.update(fallback_original=fallback,fallback_reason=reason,fallback_byte_exact=fallback and np.array_equal(proposed,original_support,equal_nan=True),parameters=PARAMETERS,topology={key:topology_audit(data,p,elements) for key,p in nodal_predictions.items()})
 if inventory(case)!=inputs or guard()[1]!=digest:raise ValueError('Gaussian inputs/freeze changed during forecast')
 out.mkdir(parents=True);np.savez_compressed(out/'predictions.npz',**predictions);np.savez_compressed(out/'nodal.npz',**nodal_predictions);write(out/'metadata.json',finite_metadata(metadata))
 write(out/'complete.json',dict(case=case,freeze_sha256=digest,inputs=inputs,artifact_sha256={key:sha(out/key) for key in ('predictions.npz','nodal.npz','metadata.json')}))
 print('Carrier Gaussian forecast committed',case['name'],'fallback',fallback,flush=True);return True

def score_all():
 frozen,digest=guard();release=read(REPORT/'phystwin-carrier-gaussian-score-release.json')
 if release.get('authorized') is not True or release.get('freeze_sha256')!=digest:raise ValueError('Explicit postforecast scoring release required')
 # Check entire fixed cohort BEFORE importing/opening any evaluator target.
 records=[validate(case) for case in frozen['cases']]
 from scripts.contact_benchmark_v12.data import load
 from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
 rows=[]
 for case in frozen['cases']:
  remaining();out=folder(case);data=load(case['spec'])
  with np.load(out/'predictions.npz') as saved:metrics={key:safe_score(data,saved[key],data['train_end'],len(data['times'])) for key in saved.files}
  row=dict(case=case,metrics=metrics,freeze_sha256=digest,prediction_sha256=sha(out/'predictions.npz'),fallback_original=read(out/'metadata.json')['fallback_original']);write(out/'scores.json',finite_metadata(row));rows.append(row)
 write(REPORT/'phystwin-carrier-gaussian-scores.json',finite_metadata(dict(cases=rows,freeze_sha256=digest,release_sha256=sha(REPORT/'phystwin-carrier-gaussian-score-release.json'),scope=PARAMETERS['scope'])))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--forecast',action='store_true');p.add_argument('--score',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.forecast:
  for name in NAMES:forecast(declared(name))
 if a.score:score_all()
