"""Manufactured shared-missed-mode counterexample; no observations or mechanics."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.special import ndtr,logsumexp
from experiments.campaign_c38_static_posterior import StaticPosterior,compare_integrators
ROOT=Path(__file__).resolve().parents[1];OP=ROOT/'runs/operator-dev/c38-shared-peak-counterexample-v01'
CONFIGS=[dict(name='single_narrow_mode',centers=[.123456],mixture=[1.]),dict(name='two_narrow_modes',centers=[.123456,.765432],mixture=[.4,.6])]
SIGMA=1e-7;LO,HI=75000.,150000.;BACKGROUND_LOG=-50.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def manufactured(config):
 centers=np.asarray(config['centers']);weights=np.asarray(config['mixture'])
 def log_likelihood(category,E):
  x=np.log(E/LO)/np.log(HI/LO);return float(logsumexp(np.r_[BACKGROUND_LOG,np.log(weights)-.5*((x-centers)/SIGMA)**2]))
 mass=np.sqrt(2*np.pi)*SIGMA*weights*(ndtr((1-centers)/SIGMA)-ndtr(-centers/SIGMA));Z=float(np.exp(BACKGROUND_LOG)+mass.sum())
 def true_cdf_unit(x):
  return float((np.exp(BACKGROUND_LOG)*x+np.sum(np.sqrt(2*np.pi)*SIGMA*weights*(ndtr((x-centers)/SIGMA)-ndtr(-centers/SIGMA))))/Z)
 return log_likelihood,true_cdf_unit,float(np.log(Z))

def diagnose(config):
 ll,truth,logZ=manufactured(config);p=StaticPosterior(ll);comparison=compare_integrators(ll);unit=np.linspace(0,1,17);actual=np.array([p.cdf(LO*2**x) for x in unit]);exact=np.array([truth(x) for x in unit]);maxreported=max(r['maximum_material_cdf_difference'] for r in comparison['comparisons'])
 return dict(config=config,sigma_unit=SIGMA,background_log_likelihood=BACKGROUND_LOG,reported_integrator_comparison=comparison,maximum_reported_cross_integrator_CDF_difference=maxreported,maximum_true_CDF_error=float(abs(actual-exact).max()),computed_log_normalizer=p.log_normalizer,analytic_log_normalizer=logZ,log_normalizer_error=abs(p.log_normalizer-logZ),shared_missed_peak_demonstrated=bool(maxreported<1e-10 and max(abs(actual-exact))>.1),unit_CDF_grid=unit.tolist(),computed_CDF=actual.tolist(),analytic_CDF=exact.tolist(),generic_helper_certified=False)

def prepare():
 if (OP/'protocol.json').exists():raise FileExistsError('Preservefrozenrecipe')
 files=['experiments/diagnose_c38_shared_peak_miss.py','tests/test_diagnose_c38_shared_peak_miss.py','experiments/campaign_c38_static_posterior.py']
 p=dict(schema='manufactured-shared-peak-counterexample-v1',status='FROZEN_BEFORE_MANUFACTURED_DIAGNOSTIC',source_sha256={f:sha(ROOT/f) for f in files},configs=CONFIGS,sigma_unit=SIGMA,background_log=BACKGROUND_LOG,scope='Source-review counterexample ONLY. No cohort/observations/physics reads, no production helper changes, no calibration or posterior admission.',expected='Adaptive and64/128Legendre can all miss same peaks. Retain their apparent agreement alongside analytic error; never promote agreement to proof.',CPU_affinity_cores=[30,31])
 write(OP/'protocol.json',p);return sha(OP/'protocol.json')

def run(digest):
 p=json.loads((OP/'protocol.json').read_text())
 if sha(OP/'protocol.json')!=digest or any(sha(ROOT/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Changedpins')
 if set(os.sched_getaffinity(0))!={30,31} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU2only')
 rows=[diagnose(c) for c in p['configs']];r=dict(completed=True,counterexample_demonstrated=all(x['shared_missed_peak_demonstrated'] for x in rows),protocol_sha256=digest,rows=rows,actual_data_or_physics_access=False,production_helper_unchanged=True)
 write(OP/'result-summary.json',r);return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--sha');x=a.parse_args();r=prepare() if x.action=='prepare' else run(x.sha);print(json.dumps(r if isinstance(r,str) else {k:r[k] for k in ['completed','counterexample_demonstrated']}))
