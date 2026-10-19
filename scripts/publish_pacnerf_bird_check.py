"""Audit one full Bird posterior before spending CPU on the larger comparison.

The readiness criteria are exploratory engineering checks on this development
scene. They are not a statistical calibration test on independent PAC scenes.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from pacnerf_comparison import write
from verify_pacnerf_discrepancy import verify_case,locate
from verify_pacnerf import verify_case as verify_baseline
from publish_pacnerf_discrepancy import snapshot_push


def readiness(summary,proof,baseline):
    new=proof['independent_prediction_scores'];old=baseline['prediction_scores']
    coverage=next(x['coverage'] for x in new['coverage_curve'] if x['nominal']==.9)
    checks={'posterior_converged':bool(summary['valid_for_ranking']),
        'raw_replay_pass':bool(proof['all_pass'] and baseline['all_pass']),
        'floor_conditioning_enabled':summary.get('settings',{}).get('model',{}).get('version')=='floor-conditioned-hierarchical-v2',
        'deep_floor_penetration_negligible':summary.get('posterior_ground_diagnostic',{}).get('all_constrained_points_fraction_below_height',{}).get('-0.02',1.)<=.001,
        'development_scene_90_coverage_at_least_80_percent':coverage>=.8,
        'development_scene_crps_no_worse_than_baseline':new['mean_crps_m']<=old['mean_crps_m']}
    return {'ready_for_larger_comparison':all(checks.values()),'checks':checks,
        'coverage90':coverage,'new_crps_m':new['mean_crps_m'],'baseline_crps_m':old['mean_crps_m'],
        'qualification':'Predeclared development-scene guardrails, not independent calibration validation. The uncertainty must cover errors without a worse proper score, and avoid deep floor penetration. Failed criteria are preserved and cannot be relabeled as success.'}


def publish(raw,inputs,baseline,output,name,push=False):
    output=Path(output)
    if output.exists():raise ValueError('Use a fresh report folder')
    output.mkdir(parents=True)
    source=locate(Path(raw)/'results-8','**/bird__4views__constant__marginal/summary.json').parent
    summary=json.loads((source/'summary.json').read_text())
    proof=verify_case(raw,inputs,8);old=verify_baseline(baseline,inputs,8)
    decision=readiness(summary,proof,old)
    for path in source.glob('*.json'):shutil.copyfile(path,output/path.name)
    write(output/'independent_verification.json',proof)
    write(output/'baseline_independent_verification.json',old)
    write(output/'readiness.json',decision)
    new=proof['independent_prediction_scores'];previous=old['prediction_scores']
    lines=['# Single Bird inference and uncertainty check','',
        'Four observations from a moving/switching camera; constant internal noise and analytically marginalized loads. Supplied geometry, appearance and the twelve-mode mechanical space remain fixed.','',
        f"Ready for larger comparison: **{decision['ready_for_larger_comparison']}**.",
        f"Convergence: Rhat {summary['worst_rhat']:.5f}; minimum bulk/tail ESS {summary['minimum_bulk_ess']:.1f}/{summary['minimum_tail_ess']:.1f}; {summary['draws_per_chain']} draws per chain.",'',
        '| Nominal interval | New coverage | Baseline coverage | New width (mm) | Baseline width (mm) |',
        '|---|---:|---:|---:|---:|']
    for a,b in zip(new['coverage_curve'],previous['coverage_curve']):
        lines.append(f"| {100*a['nominal']:.0f}% | {100*a['coverage']:.1f}% | {100*b['coverage']:.1f}% | {1000*a['mean_width_m']:.1f} | {1000*b['mean_width_m']:.1f} |")
    lines+=['',f"Mean CRPS (lower is better): new {1000*new['mean_crps_m']:.2f} mm; baseline {1000*previous['mean_crps_m']:.2f} mm.",'',decision['qualification'],'',
        'The image/physics target, parameter-dependent transport Jacobian, dense renderer, exact integration, raw quantiles, proper scores and scalar chain diagnostics were independently replayed. Repeated linear-model coverage validation is a separate mathematical check; this one deterministic Bird scene cannot establish general calibration.','',
        'The larger comparison is dispatched separately after this result is inspected. No other experimental cases start merely because this job completed.','']
    document='\n'.join(lines);(output/'RESULTS.md').write_text(document,encoding='utf-8',newline='\n')
    if push:snapshot_push(output,name,document)
    print(json.dumps(decision),flush=True)
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','inputs','baseline','output','name']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--push',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    publish(args.raw,args.inputs,args.baseline,args.output,args.name,args.push)
