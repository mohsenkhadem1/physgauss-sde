"""Reporting-only two-page appendix; requires completed independent verification."""
from pathlib import Path
import hashlib
import json
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.prefix_ridge_switch_v13.run import BASE, REPORT, read, sha, write
from scripts.prefix_ridge_switch_v13.report import summarize
from scripts.prefix_ridge_switch_v13.score import terminal_gate
from scripts.prefix_ridge_switch_v13.verify_results import guard_amendment
from scripts.coupled_contact_v13.build_surface_report import p, table, fmt
from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.lib.pagesizes import A4


def validate():
    guard_amendment()
    protocol, attempts, _ = terminal_gate()
    scores = read(REPORT / 'scores.json')
    summary = read(REPORT / 'summary.json')
    audit = read(REPORT / 'independent-audit.json')
    if audit.get('passed') is not True or audit.get('errors') or audit.get('pending'):
        raise RuntimeError('Complete passing independent result audit required')
    for key, path in [('scores_sha256', REPORT/'scores.json'),
                      ('attempts_sha256', BASE/'attempts.json'),
                      ('verifier_sha256', ROOT/'scripts/prefix_ridge_switch_v13/verify_results.py')]:
        if audit[key] != sha(path):
            raise RuntimeError('Audit evidence changed: '+key)
    digest = hashlib.sha256(json.dumps({k:v for k,v in summary.items()
        if k != 'independent_audit'}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != audit['verified_summary_metrics_sha256']:
        raise RuntimeError('Audited report metrics changed')
    rows = scores['cases']
    expected = {c['name']: c for c in protocol['cases']}
    if len(rows) != 148 or len(expected) != 148 or {r['name']:r['case'] for r in rows} != expected:
        raise RuntimeError('All 148 fixed cases required')
    for dataset in ('SoftVTBench', 'PhysTwin'):
        if summarize([r for r in rows if r['dataset']==dataset], dataset) != summary['overall_datasets'][dataset]:
            raise RuntimeError('Dataset summary mismatch')
    for cohort, datasets in summary['cohorts'].items():
        for dataset, value in datasets.items():
            if summarize([r for r in rows if r['cohort']==cohort and r['dataset']==dataset], dataset) != value:
                raise RuntimeError('Cohort summary mismatch')
    inputs = {str(path.relative_to(ROOT)):sha(path) for path in
        [REPORT/'protocol.json', REPORT/'method-freeze.json', REPORT/'scores.json',
         REPORT/'summary.json', REPORT/'independent-audit.json', BASE/'attempts.json',
         REPORT/'independent-verifier-amendment.json',
         ROOT/'scripts/coupled_contact_v13/build_surface_report.py', Path(__file__)]}
    switched = []
    for row in rows:
        if row['switched']:
            path = BASE/'forecasts'/row['cohort']/row['name']/'metadata.json'
            metadata = read(path)
            if sha(path) != attempts[row['name']]['metadata_sha256']:
                raise RuntimeError('Switched metadata changed')
            inputs[str(path.relative_to(ROOT))] = sha(path)
            switched.append((row, metadata))
    return summary, switched, inputs


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawString(42, 25, 'PhysGauss | Exploratory external benchmark appendix')
    canvas.drawRightString(553, 25, str(document.page))
    canvas.restoreState()


def main():
    summary, switched, inputs = validate()
    story = [p('Does prefix validation select better predictions?', 'Heading1'),
             p('Exploratory external benchmark appendix | Retain the prior model', 'Heading2'),
             p('We tested whether adding a guarded position-regression candidate to the existing selector improves forecasting. The completed experiment does not support promotion: SoftVT predictions are unchanged, while the PhysTwin aggregate worsens. Earlier results and forecasts remain preserved.'),
             p('Protocol and scope', 'Heading2'),
             p('All 140 SoftVTBench sequences and eight PhysTwin sequences had been examined previously. This is an exploratory algorithm ablation, not a fresh confirmation set or a transfer evaluation of the new coupled FEM solver. Initial mechanical geometry and known controller trajectories are provided; the experiment does not reconstruct these inputs from images.'),
             p('The candidate uses fixed ridge regularization 0.1, existing FEM-anchor volume repair and the same static contact approximation. Its first 70% observed prefix fits the model, and the remaining observed prefix validates it against the archived selected configuration reconstructed from independent short fits. A switch requires strictly more than 2% lower validation error. Otherwise the archived forecast is retained exactly. The original selector also used this validation interval.'),
             p('Complete-cohort prediction results', 'Heading2')]
    labels = [('ridge_switch_selected','New guarded ridge switch'),
              ('model_selection_selected','Prior selected model'),
              ('model_selection_rigid','Rigid model'),
              ('prefix_calibrated_rigid','Prefix-calibrated rigid'),
              ('position_ridge','Raw support-space ridge'),
              ('official_phystwin','Official PhysTwin baseline')]
    cells = [['Method','SoftVT mm / complete','PhysTwin mm / complete']]
    for key, label in labels:
        values = [label]
        for dataset in ('SoftVTBench', 'PhysTwin'):
            value = summary['overall_datasets'][dataset]['methods'][key]
            values.append(f"{fmt(value['complete_only_mean_mm'])} / {value['complete']}/{value['expected']}" if value['applicable'] else 'Not applicable')
        cells.append(values)
    story.append(table(cells, [215, 130, 130]))
    story.append(p('SoftVT values are coordinate RMSE; PhysTwin values are manual-track mean distance. Do not pool the two metrics. Coverage accompanies complete-case means. The raw support-space ridge differs from the guarded nodal candidate in interpolation and feasibility processing. This is not an official SoftVT policy benchmark.', 'Small'))
    story.append(PageBreak())
    story += [p('Selection decisions and limitations', 'Heading1')]
    for dataset in ('SoftVTBench','PhysTwin'):
        result = summary['overall_datasets'][dataset]
        paired = result['paired_comparisons']['model_selection_selected']
        story.append(p(f"{dataset}: {result['switched']}/{result['expected']} switched. Against the prior selected model, {paired['wins']} improve, {paired['losses']} worsen and {paired['ties']} tie across {paired['matched']} complete matched cases. Mean reduction in error: {fmt(paired['mean_gain_mm'])} mm (positive favors the extension)."))
    story.append(p('Every switched case', 'Heading2'))
    cells = [['Case','Prefix old/new mm','Future old/new mm']]
    for row, meta in switched:
        metric = 'manual_track_mean_mm' if row['dataset']=='PhysTwin' else 'coordinate_rmse_mm'
        old = row['results']['model_selection_selected']['summary'][metric]
        new = row['results']['ridge_switch_selected']['summary'][metric]
        cells.append([escape(row['name']), f"{fmt(1000*meta['old_validation_rmse_m'])} / {fmt(1000*meta['ridge_validation_rmse_m'])}", f'{fmt(old)} / {fmt(new)}'])
    story.append(table(cells, [215,130,130]))
    story.append(p('Prefix values are observed-position coordinate RMS. Future values use the dataset metric stated on page 1. They cover different intervals and, for PhysTwin, different observation metrics; their absolute magnitudes are not directly comparable.', 'Small'))
    story += [p('What this result establishes', 'Heading2'),
              p('An improvement on the observed validation interval did not reliably select a better future predictor. Geometry safeguards can preserve valid elements without producing accurate motion. These observations motivate better treatment of changing interactions; they do not establish that a particular replacement architecture or uncertainty model will solve the problem.'),
              p('What remains unchanged', 'Heading2'),
              p('We retain the prior selected model and all previous benchmark results. No threshold, candidate, or per-case choice was changed after scoring. The new results are preserved as a negative ablation. Success in the separate controlled simulation study must be assessed on its own held-out cohort and cannot substitute for external benchmark gains.'),
              p('Audit and reproducibility', 'Heading2'),
              p('All 148 forecasts were saved before evaluator access. Independent verification reconstructs ridge features, coefficients, short-prefix losses, selection decisions and new forecast metrics; archived comparator payloads are integrity-checked, not retrained. Frozen geometry/contact primitives are reused, so this is not an independent alternative physics solver. Raw datasets and prediction arrays remain local. Protocols, hashes, compact results and reproduction notes accompany this appendix.'),
              p('Evidence: reports/prefix-ridge-switch-v13/{protocol.json, scores.json, summary.json, independent-audit.json, REPRODUCE.txt}. Source: scripts/prefix_ridge_switch_v13. The historical run enforces its original deadline; a later replay requires a separately declared protocol and matching local assets.', 'Small')]
    output = ROOT/'papers/prefix-ridge-switch-v13/report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=36, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)
    if any(sha(ROOT/path)!=digest for path,digest in inputs.items()):
        raise RuntimeError('Report inputs changed during build')
    write(REPORT/'paper-build.json', dict(input_sha256=inputs,
        output=str(output.relative_to(ROOT)), output_sha256=sha(output),
        scope='Reporting-only appendix; visual inspection required before delivery'))


if __name__ == '__main__':
    main()
