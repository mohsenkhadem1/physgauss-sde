"""Build the verified campaign report from saved results; no model execution."""
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark_improvement_v14.common import BASE, REPORT, read, write, sha, check_preservation


def main():
    audits = ['softvt-b-final-independent-audit.json', 'phystwin-final-independent-audit.json',
              'phystwin-gaussian-final-independent-audit.json']
    audit_rows = {name: read(REPORT / name) for name in audits}
    if not all(row['passed'] for row in audit_rows.values()):
        raise RuntimeError('Independent audits must pass before final reporting')
    test_log = (BASE / 'finish-tests.log').read_text()
    if not re.search(r'\d+ passed', test_log) or re.search(r'\d+ failed|ERRORS', test_log):
        raise RuntimeError('Final test run has not passed')
    (REPORT / 'finish-tests.txt').write_text(test_log)
    soft = read(REPORT / 'softvt-summary-b.json')
    fresh = soft['groups']['prospective_softvt_phase_b']
    pt = read(REPORT / 'phystwin-summary.json')['groups']['all_historical_exposed']
    gaussian = read(REPORT / 'phystwin-gaussian-summary.json')
    a, b = fresh['contact_scene_state'], pt['object_frame']
    case_lines = '\n'.join(f"| {r['name']} | {r['baseline_mm']:.3f} | {r['proposal_mm']:.3f} |" for r in b['per_case'])
    checks = sum(row['checks'] for row in audit_rows.values())
    text = f'''# Verified benchmark campaign results

The campaign improved the original prediction pipeline on both datasets. SoftVT also beats the geometry-matched rigid control on the new test. PhysTwin remains behind stronger previously available controls, so its new branch is an ablation, not a replacement for the best previous benchmark result.

| Evaluation | Original | New | Change | Coverage |
|---|---:|---:|---:|---|
| SoftVT, 40 new Phase B episodes, coordinate RMSE | {a['baseline_mean_mm']:.3f} mm | {a['proposal_mean_mm']:.3f} mm | {a['relative_improvement_percent']:.2f}% lower | 40/40 episodes; 1,998/1,998 forecast frames |
| PhysTwin, 8 historically exposed cases, manual-track mean | {b['baseline_mean_mm']:.3f} mm | {b['proposal_mean_mm']:.3f} mm | {b['relative_improvement_percent']:.2f}% lower | 8/8 cases; no failed forecast frames |

SoftVT improves 36 cases and worsens 4. Its rigid control scores {fresh['rigid_scene_state']['baseline_mean_mm']:.3f} mm, so the new method improves that control by {fresh['rigid_scene_state']['relative_improvement_percent']:.2f}%. Aligned shape error is essentially unchanged: this is primarily better object motion and release prediction, not demonstrated better elasticity inference.

PhysTwin improves 3 cases and leaves 5 byte-exactly unchanged. Stronger archived controls remain better: the prior exploratory selector scores {pt['model_selection_selected']['baseline_mean_mm']:.3f} mm, position ridge {pt['position_ridge']['baseline_mean_mm']:.3f} mm, and official PhysTwin {pt['official_phystwin']['baseline_mean_mm']:.3f} mm. Those controls were not retrained in this campaign. The large unchanged single-lift-zebra error remains a major weakness.

## What changed

SoftVT estimates slip from the last eight active observed frames in the controller frame, then corrects release motion. A separate Phase B rule uses a prefix-calibrated rigid/contact fallback only when the original trajectory contains nonfinite values. It does not switch using future error. All 179 previously finite regression forecasts remain bitwise identical to Phase A. The only fallback in the 220-case evaluation repairs object-soft-task2-demo31. All 220 proposed trajectories have complete forecast coverage.

The original Phase A result is preserved: 140 historical cases improve 20.978 to 19.735 mm. Phase B also includes the earlier 40 episodes, now exposed regression cases, with 40/40 complete proposed trajectories. Their original baseline has missing frames, so its available-frame mean must not be treated as a complete paired comparison. The fresh Phase B table above has complete coverage for both methods.

PhysTwin fits the width of Gaussian attachment weights jointly with the inherited mechanics parameters. An independently shortened-prefix fit must improve validation by more than 2% before the new full-prefix prediction replaces the original. Missing or invalid comparisons retain the original exactly. Optimizer checkpoints are not all converged optima. The FEM anchor and the empirical output adapter are distinct: final adapted outputs can still invert tetrahedra. These results do not establish physical admissibility, calibrated material estimates, or reliable uncertainty.

## Other tested directions

The separately frozen, always-on Gaussian residual family scores {gaussian['comparisons']['object_frame']['proposal_mean_mm']:.3f} mm over all eight PhysTwin cases: 5 improve and 3 worsen, only a 0.74% mean gain over the original. It is not selected per case after scoring, nor combined retrospectively with the width method.

A history-aware direct-support ridge model improved prefix validation and development coordinate/shape metrics but worsened development manual tracking from 34.076 to 65.984 mm; it was rejected. The moving-carrier plus Gaussian deformation model scored 35.341 mm on that development case versus 34.076 mm originally. Its full eight-case freeze/evaluation was not completed before interruption. Its source and proposal are published as unfinished development work, not benchmark evidence. Earlier development diagnostics and rejected branches are retained. This was a multi-candidate development campaign, not a single preselected successful hypothesis.

## PhysTwin case results

| Case | Original manual mean (mm) | Width primary (mm) |
|---|---:|---:|
{case_lines}

## Videos

These are synchronized forecast-only point-cloud views with shared world axes and a moving camera, not photorealistic reconstructions. PhysTwin's left panel contains observed reconstructed points, not exact physical ground truth. Cases were fixed before scoring; numerical failures remain visibly marked.

- [SoftVT new episode: actual / original / robust](robust-videos/object-soft-task1-demo40.mp4)
- [SoftVT failure repair: actual / original / robust](robust-videos/object-soft-task2-demo31.mp4)
- [PhysTwin development lift: observed / original / width / Gaussian](phystwin-videos/single_lift_sloth.mp4)
- [PhysTwin stretch: observed / original / width / Gaussian](phystwin-videos/double_stretch_sloth.mp4)

## Verification and scope

Three independent audit implementations passed {checks:,} checks: source/input/output hashes, fixed cohort counts, regression preservation, coverage, paired summaries, and independent metric recomputation. PhysTwin raw metrics were recomputed for all eight cases; SoftVT aggregate arithmetic covers all 220 cases and raw metrics were independently checked on prespecified cases. Audit code was prepared by a separate agent; the final post-interruption execution was performed by the main agent. This is independent arithmetic/source verification, not a fresh optimizer replication. All 385 pinned historical files remain unchanged. The final test log is [finish-tests.txt](finish-tests.txt).

Inputs include initial mechanical geometry, observed-prefix 3D positions and prescribed future controller positions. No true material or force labels are supplied. SoftVT's new episodes reuse two assets and do not demonstrate independent-object OOD generalization. All eight PhysTwin cases were historically exposed; their results are exploratory. Descriptive episode bootstrap intervals should not be interpreted as independent-object confidence intervals. Metrics differ by dataset and are not pooled.

The original campaign was interrupted by a Codex usage-limit error. Background fitting, scoring and rendering finished. The user then authorized verification and publication on 1 November 2026. No fitting or new forecasts were launched during this finish step; the original campaign deadline and frozen sources were not changed. Raw datasets and prediction arrays remain local. [Reproduction instructions](REPRODUCE.txt) describe compressed reports and post-campaign audit entry points.
'''
    (REPORT / 'RESULTS.md').write_text(text)
    (REPORT / 'REPRODUCE.txt').write_text('''VERIFIED CAMPAIGN SNAPSHOT

Run from the repository root with the original local datasets and runs/ artifacts available.
Raw datasets and prediction arrays are intentionally not published. A code-only clone cannot
recompute raw metrics without acquiring those inputs. Protocol, inventory, freeze and audit
files record their hashes. Historical files and all frozen model sources remain unchanged.

Large derived JSON reports are published as deterministic .json.gz files. Consult
compressed-reports.json; run gzip -dk path/to/report.json.gz and verify its decompressed SHA-256.

Python environment: runs/phystwin-env/bin/python. Set OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 for each command.

1. python scripts/benchmark_improvement_v14/summarize_gaussian.py
2. python scripts/benchmark_improvement_v14/finish_audit.py softvt
3. python scripts/benchmark_improvement_v14/finish_audit.py phystwin
4. python scripts/benchmark_improvement_v14/finish_audit.py gaussian
5. python -m pytest -q tests/*v14.py
6. python scripts/benchmark_improvement_v14/finish_report.py

finish_audit.py authorizes only an audit with a one-hour runtime limit. It substitutes the
expired campaign time check in process memory, while all source hashes and forecast/input
checks remain enforced. It cannot launch a fit or forecast. Do not edit frozen common.py,
configuration, protocols or model sources to resume an expired campaign.

The report builder requires successful audit JSON files and the saved finish-tests.log.
Videos have frame-count/fps/full-decode checks and first/middle/last visual review; the visual
review is not a claim that a human watched every frame. Predictor outputs are never changed
by rendering. See each video directory's provenance.json, machine-qa.json and visual-qa.json.

The carrier Gaussian freeze-proposal is unfinished development evidence. It is not an
authorized final freeze or a completed benchmark run. No post-score family selection occurs.
''')
    for directory in ['robust-videos', 'phystwin-videos']:
        qa = read(REPORT / directory / 'machine-qa.json')
        if not qa['passed']:
            raise RuntimeError('Video decode QA failed')
        write(REPORT / directory / 'visual-qa.json', dict(
            passed=True, scope='Main agent inspected first, middle and last frame contact sheets; labels, shared axes, failures and geometry visible. Not every frame watched.',
            machine_qa_sha256=sha(REPORT / directory / 'machine-qa.json'),
            contact_sheets={clip['review_sheet']: clip['sheet_sha256'] for clip in qa['clips']}))
    write(REPORT / 'finish-summary.json', dict(
        completed_scope='Post-campaign verification, video review and publication; no new model execution',
        preserved_historical_files=check_preservation(), audit_checks=checks,
        audits={name: dict(passed=row['passed'], checks=row['checks'], sha256=sha(REPORT/name)) for name,row in audit_rows.items()},
        softvt_fresh40=a, phystwin_width=b,
        phystwin_status='Improves original baseline, still worse than stronger archived controls; not promoted over prior best',
        remaining_scientific_work='Better PhysTwin generalization, physically admissible deformation and calibrated inference remain unresolved. Carrier full benchmark unfinished.'))
    print('Verified report written;', checks, 'independent audit checks')


if __name__ == '__main__':
    main()
