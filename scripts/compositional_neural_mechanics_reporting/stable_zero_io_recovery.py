"""Directory-only replay of frozen stable controls; previous96 writes failed.

Checkpoints/rule were frozen before reference scoring. Successful artifacts are
created only after this I/O correction, when primary outcomes are already visible.
No retraining, parameter choice or preprocessing change is made here.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics import stable_zero_evaluate as original
from scripts.compositional_neural_mechanics.dataset_runner import npz as original_npz,write
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=original.ROOT;R=original.REPORT;OUT=ROOT/'runs/compositional-neural-mechanics-v16/stable-zero-io-recovery';PLAN=R/'stable-zero-io-recovery-plan.json'

def mkdir_npz(path,**arrays):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 original_npz(path,**arrays)

def implementation():
 spec=importlib.util.spec_from_file_location('_stable_zero_directory_only',original.__file__);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.OUT=OUT;module.COMMIT=OUT/'predictions-commit.json';module.npz=mkdir_npz
 return module

def prepare():
 if PLAN.exists()or OUT.exists():raise ValueError('Preserve previous recovery')
 original.guard();prior=json.loads(original.COMMIT.read_text())
 if len(prior['rows'])!=96 or any(r['state']!='failed'or'FileNotFoundError'not in r.get('error','')for r in prior['rows']):raise ValueError('Expected preserved96write failures')
 p=dict(scope=__doc__,rule='Only create destination parents before calling exact original atomicnpz writer; private clonedmodule changes outputnamespace andwriter only.',source_sha256=closure([Path(__file__).resolve(),ROOT/'tests/test_stable_zero_io_recovery_v16.py']),old_commit_sha256=sha(original.COMMIT),control_plan_sha256=sha(original.PLAN),runtime=dict(python=str(Path(sys.executable).absolute()),torch=torch.__version__,numpy=np.__version__),expected=96,model_selection='None; both frozen6000/30000stepvariants retained.',successful_predictions_timing='After directoryfix and visible primaryoutcomes; never claim successfulpretruthforecastcommit.')
 write(PLAN,p)

def guard():
 p=json.loads(PLAN.read_text());original.guard()
 if sha(original.COMMIT)!=p['old_commit_sha256']or sha(original.PLAN)!=p['control_plan_sha256']:raise ValueError('Original evidence changed')
 if p['runtime']!=dict(python=str(Path(sys.executable).absolute()),torch=torch.__version__,numpy=np.__version__):raise ValueError('Exact original controlCPU runtime required')
 for path,h in p['source_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('I/O recovery source changed')
 return p

def run():
 guard();module=implementation();module.predict();commit=json.loads(module.COMMIT.read_text());plan,records=module.runtime_records();variants=module.guard()['variants'];expected={(r['scene']['name'],r['role'],a['name'],m)for r in records for a in r['actions']for m in variants}
 if len(commit['rows'])!=96 or {(r['scene_id'],r['cohort'],r['action'],r['method'])for r in commit['rows']}!=expected:raise ValueError('Exact96 identities required')
 for row in commit['rows']:
  if row['state']!='complete':raise ValueError('Recovery must produce96successful artifacts: '+row.get('error',''))
  if sha(ROOT/row['artifact'])!=row['sha256']:raise ValueError('Written artifact hash mismatch')
  with np.load(ROOT/row['artifact'],allow_pickle=False)as data:
   module.validate_value(data,plan['material_query'])
   if not np.array_equal(data['material_query'],np.asarray(plan['material_query'])):raise ValueError('Query mismatch')
 guard();write(R/'stable-zero-io-recovery-prediction-audit.json',dict(passed=True,expected=96,successful=96,finite=96,plan_sha256=sha(PLAN),commit_sha256=sha(module.COMMIT),old_failed_commit_preserved_sha256=sha(original.COMMIT),scope=__doc__))
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();prepare()if a.prepare else run()
