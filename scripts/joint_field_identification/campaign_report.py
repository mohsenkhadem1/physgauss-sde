"""Write a concise completed-campaign report from audited aggregate evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text())
def main():
 base='reports/joint-field-surface-v04/'
 s=read(base+'summary.json'); c=read(base+'continuous-summary.json'); ca=read(base+'continuous-independent-final-audit.json')
 assert s['complete'] and c['complete'] and ca['integrity_pass']
 h=read('reports/joint-field-hierarchy-surface-v05/summary.json');t=read('reports/joint-field-temporal-v06/summary.json')
 g=lambda policy,step=4:next(r for r in s['summary'] if r['population']=='matched' and r['policy']==policy and r['step']==step)
 hg=lambda condition:next(r for r in h['means'] if r['model']=='hierarchical' and r['condition']==condition)
 cg=lambda stage,model='continuous-smc':next(r for r in c['summary'] if r['population']=='continuous-coarse' and r['stage']==stage and r['model']==model)
 tg=lambda model:next(r for r in t['means'] if r['model']==model and r['condition']=='views')
 a,f,v,i,p=g('active'),g('fixed'),g('fixed-view'),g('active',0),g('fixed-probe')
 hi,hv,hp=hg('initial'),hg('views'),hg('probes');ct,cf=cg('active'),cg('fixed');tf,ta=tg('scale-correlation'),tg('amplitude-only')
 ph=read(base+'continuous/posthoc4096/diagnosis.json');pair=next(r for r in ph['comparisons'] if '-n4096-' in r['first'] and '-n4096-' in r['second'])
 text=f'''JOINT MATERIAL–INTERACTION IDENTIFICATION — COMPLETED CAMPAIGN

WHAT CHANGED
We now infer a distribution of spatial material, compliant support and external
load fields from sparse visible-surface movement. Temporal forces are integrated
out conditionally rather than replaced by one fitted force history. Predictions
retain force/material dependence, and expected predictive improvement compares
additional camera tracks with calibrated reset probes. FEM supplies the forward
response; it is not the claimed contribution. The earlier marginalized-force
cube and camera/probe beam prototypes remain acknowledged predecessors.

MEASURED RESULTS
Primary surface study: 320 test cases across two fixed geometries, including128
matched cases and three64-case mismatch populations. On the128matched cases:
- Combined coordinate RMSE: {i['rmse_mm']:.5f}mm initially, {a['rmse_mm']:.5f}mm after four adaptive scalars.
- Fixed camera-only: {v['rmse_mm']:.5f}mm; adaptive improves {(1-a['rmse_mm']/v['rmse_mm'])*100:.1f}%.
- Fixed mixed: {f['rmse_mm']:.5f}mm; adaptive improves {(1-a['rmse_mm']/f['rmse_mm'])*100:.1f}%.
- New-load prediction alone: active {a['changed_load_rmse_mm']:.5f}mm, probe-only {p['changed_load_rmse_mm']:.5f}mm.
  Thus adaptive wins on the declared combined target, not every endpoint.
- Nominal90% prediction coverage after adaptive evidence: {100*a['coverage90']:.2f}%.

Fresh128-case uncertain-force-scale study:
- Four camera measurements reduce hidden-motion error by {(1-hv['hidden_rmse_mm']/hi['hidden_rmse_mm'])*100:.1f}%.
- Four calibrated probes reduce new-load error by {(1-hp['changed_load_rmse_mm']/hi['changed_load_rmse_mm'])*100:.1f}%
  and material-field error by {(1-hp['material_relative_rmse']/hi['material_relative_rmse'])*100:.1f}%.
- Post-probe force90% interval coverage: {100*hp['force_coverage90']:.2f}%.
These are separate four-measurement branches from the same initial observations.

Continuous inference on64continuous-prior cases:
- Active combined RMSE {ct['rmse_mm']:.5f}mm versus fixed mixed {cf['rmse_mm']:.5f}mm.
- Active prediction90% coverage {100*ct['coverage90']:.2f}%; force coverage {100*ct['force_coverage90']:.2f}%.
- {c['completed']}fits complete:192main plus96seed/particle diagnostics.
- {c['flagged_comparisons']}/{c['comparison_count']}descriptive numerical comparisons exceed the frozen thresholds.
No threshold flag is a convergence proof. Continuous inference reuses finite-
planner actions. Different finite/continuous priors prevent treating their
comparison as pure quadrature error.
A targeted posthoc follow-up of the flagged block case repeats inference with
two independent4096-particle runs. Their maximum parameter-CDF difference is
{pair['max_cdf']:.5f}, and target-mean RMS difference is {pair['prediction_mean_rms_mm']:.5f}mm.
The original flags remain reported; this is not a global convergence certificate.

Fresh64-case temporal-prior follow-up:
After four fixed views, mixing temporal correlation and amplitude reduces hidden
error from {ta['hidden_rmse_mm']:.5f} to {tf['hidden_rmse_mm']:.5f}mm. Force90%coverage rises from
{100*ta['force_coverage90']:.2f}% to {100*tf['force_coverage90']:.2f}%, still below90%. Material error worsens.
This targeted posthoc repair preserves the original stress-test failure.

ADDITIONAL CHECKS
Actual area/time-integrated surface traction, interval dependence, mechanical
normalization, surface visibility, strain, paired numerical refinement, independent
Gaussian identities and all aggregate statistics are audited in the reports.
The separate32-case finer inverse study uses interior sensors; it is not a camera
benchmark. Historical interior-sensor versions are retained, never relabeled.

SCOPE
Inputs currently require known geometry, camera calibration and correspondences.
The new model is gravity-free linear-elastic quasistatic response with signed
tractions and bilateral compliant supports. It does not yet infer these fields
end-to-end from raw RGB, discover unilateral contact, or forecast impacts.
The Gaussian spatial family is deliberately low-dimensional. Matched synthetic
coverage does not establish real-video physical calibration. Prior mismatch and
numerical discretization can materially reduce coverage.
Earlier public-data rendering, contact and boundary studies are preserved in the
unchanged archived supplement with their original adapters and limitations.
They are not attributed to this new joint posterior.

DELIVERABLES
papers/cvpr2027-joint-identification/main.pdf
papers/cvpr2027-joint-identification/supplement.pdf
papers/cvpr2027-joint-identification/release/latex-source.zip
docs/JOINT_FIELD_IDENTIFICATION.md
output/joint-field-surface-v04/ (figures and true-scale quasistatic replay videos)
Compact complete-case evidence is available as CSV.gz; raw banks stay local.
'''
 (ROOT/base/'REPORT.txt').write_text(text)
 print('Completed campaign report written')
if __name__=='__main__':main()
