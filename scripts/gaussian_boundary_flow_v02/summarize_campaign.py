"""Compact evidence index; never replaces or changes per-case score artifacts."""
import json,hashlib
from pathlib import Path

ROOT=Path('reports/gaussian-boundary-flow-v02')

def compact(value):
    if isinstance(value,dict):return {k:compact(v)for k,v in value.items()if k not in ('frames','rows','cases','per_case','per_case_errors','per_world')}
    if isinstance(value,list) and len(value)>20:return {'local_entries':len(value),'omitted_from_compact_index':True}
    if isinstance(value,list):return [compact(v)for v in value]
    return value


def summarize():
    result={'scope':'Audited development campaign; historical external cases, not a replacement for prior best benchmark results',
      'status':'Active; broader evaluation is gated on demonstrated predictive improvement',
      'datasets':{'SoftVTBench':'600 local episodes / 3 assets; 220 historical development episodes; fresh arrays not opened',
      'PhysTwin':'22 historically exposed real sequences; no new sealed external test',
      'Isaac':'Existing 272 episodes and contact/boundary studies retained',
      'PokeFlex':'No approved local raw recordings; not tested',
      'Deform360':'Paused at user request'},'evidence':{}}
    names=['matched-results.json','full-covariance-coverage-comparison.json','reference-posterior/summary.json',
      'release-safeguarded-v3-qualification.json','affine-observation-audit.json','tests-summary.json',
      'softvt-package-pilot-scores.json','push-sloth-package-pilot-scores.json',
      'softvt-affine-pilot-scores.json','push-sloth-affine-pilot-scores.json',
      'retrieval-comparison.json','pose-comparison.json','pose-contact-comparison.json']
    for name in names:
        p=ROOT/name
        if p.exists():result['evidence'][name]={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'summary':compact(json.loads(p.read_text()))}
    (ROOT/'evidence-index.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[]
    for label,filename in [('SoftVT original','softvt-package-pilot-scores.json'),('PhysTwin push original','push-sloth-package-pilot-scores.json'),('SoftVT corrected','softvt-affine-pilot-scores.json'),('PhysTwin push corrected','push-sloth-affine-pilot-scores.json'),('SoftVT restored transport/contact','pose-contact-softvt-scores.json'),('PhysTwin push restored transport/contact','pose-contact-push-sloth-scores.json')]:
        p=ROOT/filename
        if not p.exists():continue
        d=json.loads(p.read_text());rows.append('\n'+label+' (one exposed development sequence)')
        for name,data in d['methods'].items():
            if 'incoming_v22' in data:data=data['incoming_v22']
            q=data['summary'];rows.append(f"  {name}: coordinate {q['coordinate_rmse_mm']:.3f} mm; rigid-aligned shape {q['aligned_shape_rmse_mm']:.3f} mm; {q['failed_frames']} failed frames")
    text='''Audited Gaussian-boundary campaign — development checkpoint

Controlled tests show useful prediction improvements. External benchmark-wide
improvement is NOT established. Previous best models and simulation results remain
intact. Three independent agents audited scientific validity, numerical behavior
and dataset/evaluation protocols, and their findings changed implementation choices.

Completed improvements
- Corrected duplicated training noise; compared capacity-matched probabilistic
  alternatives. The simple mixture predicts better on the controlled block tests.
- Added informative deformation rather than relying on nearly rigid motion.
- Validated a small conditional reference posterior with explicit refinement.
- Separated raw mechanical states from tracking/rebased output.
- Added affine-complete observation mapping, correlated temporal observation noise,
  bound PCA/model checkpoints, and discarded-feature mismatch screening.
- Qualified two finite-range Gaussian grasp release examples. Impact convergence
  remains unresolved; this is not a learned contact/impact benchmark result.

External diagnostic results (lower error is better)
'''+ '\n'.join(rows)+'''

Interpretation
Position prediction and deformation prediction are different. A method can beat
persistence by moving the object while predicting its shape poorly. Neither good
rendering nor plausible motion establishes correct boundary/material inference.
Small pilots use 64 nodes and 12 selected frames; archived stronger methods use
larger budgets. Matched evaluation points do not make training budgets matched.

The new pipeline omitted old pose/release/contact capabilities. A separately frozen
restored transport/contact baseline reduces SoftVT raw flow error 56.22 to 18.31 mm.
The pushed toy still loses to velocity and retains large shape error. Keep this
useful baseline alongside raw mechanics, not as proof of recovered physical forces.
Floor projection is an approximation; pose uncertainty and inferred slip are absent.
Expand the frozen comparison across exposed cases before any fresh confirmation.
Do not select a new model over preserved best results merely because it is newer.

Full provenance, audits and commands: evidence-index.json, decisions.json,
campaign-review.txt, retrieval-comparison.txt and scripts/gaussian_boundary_flow_v02.
Raw datasets, checkpoints, sampled trajectories and large per-frame reports stay local.
'''
    (ROOT/'RESULTS.txt').write_text(text)
    print(json.dumps({'evidence_sources':len(result['evidence']),'compact_bytes':(ROOT/'evidence-index.json').stat().st_size}))

if __name__=='__main__':summarize()
