"""Generate the complete campaign narrative from sealed study summaries.

This report does not certify publication or replace independent scientific and
visual audits. Missing bowl results prevent a partial report being called final.
"""
from pathlib import Path
import json,math,hashlib

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/gap-first-v01'


def read(relative):return json.loads((REPORT/relative).read_text())


def validate_completion():
    """Completeness/provenance gate only; does not require the later release audit."""
    final=read('bowl/finalization.json')
    if final.get('status')!='results-audits-and-figures-complete':
        raise SystemExit('Bowl finalization is not complete; no final report/package generated')
    summary_path=REPORT/'bowl/summary.json'
    if hashlib.sha256(summary_path.read_bytes()).hexdigest()!=final['summary_sha256']:
        raise SystemExit('Bowl summary differs from completed finalizer record')
    protocol=read('bowl/protocol.json')
    if len(protocol['hypotheses'])!=16 or protocol['noise_replicates_per_material']!=2:
        raise SystemExit('Bowl protocol dimensions changed; review release expectations')
    for relative,digest in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise SystemExit(f'Frozen bowl source changed: {relative}')
    bowl=read('bowl/summary.json'); rows=read('bowl/case-results.json')
    expected={(model,cohort,world,rep,stage)
      for model in ('fem','resolution-mixture')
      for cohort in ('matched','mesh-mismatch') for world in range(16) for rep in range(2)
      for stage in (['initial','fixed_push','fixed_tilt','active']+
                    (['shared_coarse_active'] if model=='resolution-mixture' else []))}
    keys=[(r['model'],r['cohort'],r['hypothesis'],r['replicate'],r['stage']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected or bowl['rows']!=len(expected):
        raise SystemExit('Bowl cohort incomplete/duplicated: expected all 576 frozen records')
    if bowl['source_sha256']!=hashlib.sha256((REPORT/'bowl/case-results.json').read_bytes()).hexdigest():
        raise SystemExit('Bowl aggregate source hash mismatch')
    expected_groups={f'{m}/{c}/{s}' for m,c,w,r,s in expected}
    if set(bowl['groups'])!=expected_groups:
        raise SystemExit('Bowl summary group set mismatch')
    visibility=read('bowl/all-state-geometry-audit.json')
    if visibility['completed_files']!=96 or visibility['expected_files']!=96 or len(visibility['files'])!=96:
        raise SystemExit('Bowl geometry audit must cover all 2x16x3 trajectory files')
    seal=read('bowl/prediction-seal.json')
    if seal['sha256']!=hashlib.sha256((REPORT/'bowl/prospective-predictions.json').read_bytes()).hexdigest():
        raise SystemExit('Bowl prospective prediction seal mismatch')
    stem=read('stem/summary.json')
    if set(stem['conditions'])!={'repeat_tip','visible_base','concealed_base'}:
        raise SystemExit('Stem conditions incomplete')
    for policies in stem['conditions'].values():
        if len(policies)!=8 or any(v['n']!=96 for v in policies.values()):
            raise SystemExit('Stem policy/cohort count mismatch')
    protocol=read('drop/protocol.json');assets=set(protocol['assets'])
    status=read('drop/supervisor.json')
    expected_runs={a+'/'+c for a in assets for c in protocol['cases']}
    if status.get('status')!='complete' or set(status['completed'])!=expected_runs or status['failures']:
        raise SystemExit('Primary native drop cohort incomplete')
    paths=list((REPORT/'drop').glob('*-analysis.json'))
    if {json.loads(p.read_text())['asset'] for p in paths}!=assets or len(paths)!=3:
        raise SystemExit('Native drop analysis asset set mismatch')
    native=read('drop/native-reference-extension/summary.json')
    if not native.get('complete') or native['assets']!=3 or len(native['results'])!=3 or {r['asset'] for r in native['results']}!=assets:
        raise SystemExit('Native extension incomplete')
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):
            raise SystemExit('Nonfinite value in final scientific results')
        if isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    for value in (stem,bowl,rows,native):finite(value)
    return dict(bowl_records=len(rows),bowl_trajectory_files=96,drop_cases=len(expected_runs),stem_physical_cases=96)


def main():
    validate_completion()
    stem=read('stem/summary.json');bowl=read('bowl/summary.json')
    visibility=read('bowl/all-state-geometry-audit.json')
    if visibility['completed_files']!=visibility['expected_files']:
        raise SystemExit('Bowl geometry/visibility audit is incomplete')
    paths=sorted((REPORT/'drop').glob('*-analysis.json'))
    if len(paths)!=3:raise SystemExit('Expected all three predeclared drop assets')
    native=read('drop/native-reference-extension/summary.json')
    if not native['complete']:raise SystemExit('Native-predictor extension incomplete')
    lines=['GAP-FIRST CAMPAIGN: COMPLETE SCIENTIFIC RESULTS',
      'Question: does uncertainty about the object matter for its next specified interaction?',
      '', 'SCOPE',
      'Three separate studies: native ReconPhys drop, compliant stem, and nonlinear bowl contact.',
      'The user explicitly added the bowl after initial goal creation. Raw datasets remain local.',
      'These studies do not constitute a newly trained end-to-end raw-video model.',
      'Earlier manuscript, supplement, spatial-field inference and public-data results are preserved.',
      '', '1. NATIVE DROP',
      'Three lexicographically selected released assets; 27 primary native runs.',
      'Same geometry, anchors, initial state and friction; (m,k,d)=(1,300,1) or (2,600,2).',
      'Force is a known uniform body force per anchor, not a localized contact poke.',
      'Coordinates and spring parameters are native normalized units, not calibrated SI material values.',
      'Passive PSNR below pools pixel MSE over the four frozen cameras and all frames.',
      'Changed-response error is 21-coordinate centroid RMS over frames 18–24.',
      'asset | passive PSNR dB | changed-response gap | MAP mean error before -> after']
    for path in paths:
        d=json.loads(path.read_text());mse=sum(v['pixel_mse'] for v in d['passive_pixel_comparison'])/4
        psnr=10*math.log10(255**2/mse) if mse else math.inf
        stages={r['stage']:r for r in d['decision_summary']}
        lines.append(f"{d['asset']} | {psnr:.4f} | {d['changed_force_target_pair_rms_difference']:.8f} | "
            f"{stages['before']['mean_realized_rms_error']:.8f} -> {stages['after']['mean_realized_rms_error']:.8f}")
    lines += ['', 'Each world has 100 observation-noise repeats: 600 trials across three assets, not 600 objects.',
      'The same-evidence MAP estimator has identical corrected point accuracy; no unique accuracy win.',
      'The separately frozen native-network extension scores a different, unchanged point prediction.',
      'It has no native-success cases; it cannot establish risk discrimination over all operating conditions.']
    for r in native['results']:
        before=next(z for z in r['risk_summary'] if z['stage']=='before')
        lines.append(f"Native {r['asset']}: mean coordinate RMS={before['mean_native_rms']:.8f}, "
          f"predicted failure={before['mean_native_risk']:.6f}, realized failure={before['native_failure_rate']:.6f}.")
    lines += ['', 'The imperfect released-ground-truth-parameter replay prevents assigning all recorded-video error to the predictor.',
      'Constructed-sequence PSNR is view-0 pooled MSE; released-video controls use mean per-frame four-view PSNR.',
      'Final illustration cameras fit all states and differ from the frozen quantitative cameras.',
      '', '2. COMPLIANT STEM',
      '96 independent physical cases, each in three paired task/access conditions; 288 records.',
      'Known force, geometry and calibrated synthetic sensor noise. Independent beam FEM verification.',
      'Tip-only observations constrain bending+base compliance; corrective load locations exclude the held-out 0.75L load.',
      'Risk and action choices were saved before held-out responses were evaluated.',
      'The next table reports across-case scalar displacement RMSE, not a mean of vector RMS values.',
      'condition | policy | point predictor | RMSE mm | failure rate | predicted failure | Brier | action counts']
    for condition,policies in stem['conditions'].items():
        for policy,value in policies.items():
            for model,m in value['models'].items():
                lines.append(f"{condition} | {policy} | {model} | {m['rmse_mm']:.6f} | "
                    f"{m['observed_failure_rate']:.6f} | {m['predicted_failure_rate']:.6f} | "
                    f"{m['brier_score']:.6f} | {json.dumps(value['actions'],sort_keys=True)}")
    lines += ['', 'Paired decision-loss differences: full posterior policy minus control.',
      'Loss = threshold failure + declared observation cost; bootstrap clusters by 96 physical cases.']
    for control,r in stem['paired_decision_loss'].items():
        lines.append(f"{control}: {r['mean_failure_plus_cost_difference']:.8f}, 95% interval {r['paired_case_bootstrap95']}")
    lines += ['There is no convincing advantage over the strong cost-aware Gaussian moment-risk baseline.',
      'Some fixed controls have lower RMSE; adaptive selection trades accuracy against measurement cost.',
      'A fresh-workspace reproduction exactly matches the complete stem summary.',
      '', '3. NONLINEAR BOWL',
      'One nonbox geometry and 16 material/contact parameter worlds, with two noise repeats per world.',
      'Coarse and finer settings are paired numerical variants of those worlds, not additional independent objects.',
      'Neo-Hookean dynamics, gravity, concave bowl penalty contact and regularized friction.',
      'Exactly one position-controlled push or bowl-rotation calibration episode; no proceed/cost claim here.',
      'The stronger transverse held-out push is separate from calibration. No reaction force is observed during fitting.',
      'The finer-space/time simulation is a numerical-model stress reference, not certified continuum truth.',
      'The secondary resolution mixture includes that finer model by construction with equal resolution prior.',
      'Motion entries are means of per-case point-Euclidean RMS in mm.',
      'Forces are separate bowl/tool vector RMS in N after a 20 ms trailing boxcar average.',
      'model/cohort/stage | motion MAP RMS mm | predicted failure | actual failure | 90% coordinate coverage | bowl force RMS N | tool force RMS N']
    for key,g in bowl['groups'].items():
        fields=['map_rms_mm','predicted_failure_probability','actual_failure','coordinate_coverage90',
                'bowl_reaction_map_vector_rms_N','tool_reaction_map_vector_rms_N']
        lines.append(key+' | '+' | '.join(f"{g[field]['mean']:.7f}" for field in fields))
    lines += ['', f"All-state geometry audit: {visibility['completed_files']} trajectory files; "
      f"{visibility['total_visibility_checks']} visibility checks; {visibility['total_visibility_failures']} failures."]
    if visibility['total_visibility_failures']:
        lines.append('Camera-visible interpretation is not established for every declared observation. All failing cases are retained; inspect the audit before interpreting scores.')
    lines += ['Full paired cluster intervals, action counts, force coverage and interval widths are in bowl/summary.json.',
      'Numerical failures and the failed adapted spring-core qualification are retained; neither is a native PhysTwin accuracy ranking.',
      '', 'REPRODUCTION AND DELIVERY',
      'See README.txt and drop/README.txt, bowl/README.txt for commands, sources, assumptions and raw-data policy.',
      'Video/figure directories: output/gap-first-v01/drop, stem and bowl.',
      'Paper addendum and editable source: papers/cvpr2027-joint-identification/.',
      'The final completion/publication audit is separate from this scientific report.',
      'All results remain conditional on the declared physical, numerical, visibility and noise models.']
    (REPORT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('Complete scientific report written; publication and PDF checks remain separate.')


if __name__=='__main__':main()
