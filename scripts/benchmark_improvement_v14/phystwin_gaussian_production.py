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
from scripts.benchmark_improvement_v14.phystwin_gaussian_residual import predict,SETTINGS
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.experiment import finite_metadata
from scripts.contact_benchmark_v12.feasibility import inherited_elements
from scripts.contact_benchmark_v12.feasible_experiment import topology_audit
FREEZE=REPORT/'phystwin-gaussian-freeze.json'
PARAMETERS=dict(SETTINGS,pose_strength=.5,spatial_basis='Normalized rest-node Gaussian; observed basis=observation_matrix@nodal_basis',features='Legacy controller-local displacement means plus rotated gravity; prefix standardization',intercept_regularized=True,active_frames_only=True,initial_frame='Inherited FEM initial frame',policy='Always-on same family for all eight; only input/numerical/nonfinite failure retains exact original object_frame',scope='Globally development-selected exploratory family; SHORT validation slightly worse; no independent confirmation or physical-admissibility claim')

def folder(case):return BASE/'gaussian-evaluation'/case['cohort']/case['name']

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
 for path in (Path(__file__),ROOT/'scripts/benchmark_improvement_v14/phystwin_gaussian_residual.py',ROOT/'scripts/benchmark_improvement_v14/phystwin_gaussian_test.py'):
  sources[str(path.relative_to(ROOT))]=sha(path)
 proposal=dict(schema_version=1,protocol_sha256=sha(REPORT/'protocol.json'),parent_width_freeze_sha256=sha(REPORT/'phystwin-freeze.json'),parameters=PARAMETERS,cases=cases,source_sha256=sources,input_sha256={case['name']:inventory(case) for case in cases},development_evidence_sha256={str(path.relative_to(ROOT)):sha(path) for path in (REPORT/'phystwin-gaussian-plan.json',REPORT/'phystwin-gaussian-development.json')})
 write(REPORT/'phystwin-gaussian-freeze-proposal.json',proposal);return proposal

def guard():
 remaining();frozen=read(FREEZE)
 if frozen['protocol_sha256']!=sha(REPORT/'protocol.json') or frozen['parameters']!=PARAMETERS:raise ValueError('Gaussian protocol/settings changed')
 if frozen['parent_width_freeze_sha256']!=sha(REPORT/'phystwin-freeze.json'):raise ValueError('Parent width freeze changed')
 cases=[declared(name) for name in NAMES]
 if frozen['cases']!=cases or set(frozen['input_sha256'])!=set(NAMES):raise ValueError('Gaussian eight-case cohort changed')
 for path,digest in frozen['source_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Gaussian frozen source changed: '+path)
 for path,digest in frozen['development_evidence_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Gaussian development evidence changed')
 required={'scripts/benchmark_improvement_v14/phystwin_gaussian_residual.py','scripts/benchmark_improvement_v14/phystwin_gaussian_production.py','scripts/contact_benchmark_v12/object_frame.py','scripts/external_benchmark/data.py'}
 if not required.issubset(frozen['source_sha256']):raise ValueError('Missing Gaussian source freeze coverage')
 return frozen,sha(FREEZE)

def validate(case):
 frozen,digest=guard();out=folder(case);record=read(out/'complete.json')
 if record['case']!=case or record['freeze_sha256']!=digest or record['inputs']!=frozen['input_sha256'][case['name']]:raise ValueError('Gaussian forecast identity/contract changed')
 if inventory(case)!=record['inputs']:raise ValueError('Gaussian forecast inputs changed')
 for key in ('predictions.npz','metadata.json'):
  if sha(out/key)!=record['artifact_sha256'][key]:raise ValueError('Gaussian forecast artifact changed')
 return record

def forecast(case):
 case=declared(case['name']);frozen,digest=guard();out=folder(case)
 if (out/'complete.json').exists():validate(case);return True
 if out.exists():raise ValueError('Do not silently overwrite partial Gaussian forecast')
 inputs=inventory(case)
 if inputs!=frozen['input_sha256'][case['name']]:raise ValueError('Gaussian frozen inputs changed')
 original_path=ROOT/'runs/contact-benchmark-v12/evaluation'/case['cohort']/case['name']/'predictions.npz'
 with np.load(original_path) as saved:
  original=saved['object_frame'].copy();inherited=saved['v11_fem'].copy()
 data=load_payload(ROOT/'runs/prefix-ridge-switch-v13/inputs'/case['name']/'full.npz')
 with np.load(anchor_folder(case)/'prediction.npz') as saved:anchor=saved['predicted'].copy()
 if not np.array_equal(anchor,inherited,equal_nan=True):raise ValueError('Original full FEM source differs from pinned original archive')
 fallback=False;reason=None
 try:
  strict(data,case['name'])
  if anchor.shape!=original.shape or not np.isfinite(anchor).all():raise ValueError('Inherited full FEM anchor is invalid')
  proposed,metadata=predict(data,anchor)
  if proposed.shape!=original.shape or not np.isfinite(proposed).all():raise ValueError('Gaussian prediction is nonfinite or wrong shape')
 except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
  fallback=True;reason=str(error);proposed=original.copy();metadata=dict(failed_candidate=True,reason=reason)
 predictions=dict(object_frame=original,gaussian_v14=proposed,inherited_fem=anchor)
 elements=inherited_elements(data)
 metadata.update(fallback_original=fallback,fallback_reason=reason,fallback_byte_exact=fallback and np.array_equal(proposed,original,equal_nan=True),parameters=PARAMETERS,topology={key:topology_audit(data,p,elements) for key,p in predictions.items()})
 if inventory(case)!=inputs or guard()[1]!=digest:raise ValueError('Gaussian inputs/freeze changed during forecast')
 out.mkdir(parents=True);np.savez_compressed(out/'predictions.npz',**predictions);write(out/'metadata.json',finite_metadata(metadata))
 write(out/'complete.json',dict(case=case,freeze_sha256=digest,inputs=inputs,artifact_sha256={key:sha(out/key) for key in ('predictions.npz','metadata.json')}))
 print('Gaussian forecast committed',case['name'],'fallback',fallback,flush=True);return True

def score_all():
 frozen,digest=guard();release=read(REPORT/'phystwin-gaussian-score-release.json')
 if release.get('authorized') is not True or release.get('freeze_sha256')!=digest:raise ValueError('Explicit postforecast scoring release required')
 # Check entire fixed cohort BEFORE importing/opening any evaluator target.
 records=[validate(case) for case in frozen['cases']]
 from scripts.contact_benchmark_v12.data import load
 from scripts.external_benchmark.evaluate import expand_prediction
 from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
 rows=[]
 for case in frozen['cases']:
  remaining();out=folder(case);data=load(case['spec'])
  with np.load(out/'predictions.npz') as saved:metrics={key:safe_score(data,expand_prediction(data,saved[key]),data['train_end'],len(data['times'])) for key in saved.files}
  row=dict(case=case,metrics=metrics,freeze_sha256=digest,prediction_sha256=sha(out/'predictions.npz'),fallback_original=read(out/'metadata.json')['fallback_original']);write(out/'scores.json',finite_metadata(row));rows.append(row)
 write(REPORT/'phystwin-gaussian-scores.json',finite_metadata(dict(cases=rows,freeze_sha256=digest,release_sha256=sha(REPORT/'phystwin-gaussian-score-release.json'),scope=PARAMETERS['scope'])))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--forecast',action='store_true');p.add_argument('--score',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.forecast:
  for name in NAMES:forecast(declared(name))
 if a.score:score_all()
