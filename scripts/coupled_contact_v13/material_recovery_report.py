"""Score selected FEM parameters after fitting; never alter selection or forecasts."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.report_directory)
    protocol_path = directory/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    sources = {str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    splits = {}
    for split in ('development', 'validation', 'reserved'):
        path = directory/f'{split}-results.json'
        if not path.exists():
            continue
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        rows = []
        for obj in protocol['objects']:
            if obj['split'] != split:
                continue
            selected = (data.get('selections', {}).get(obj['id']) or {}).get('fem', {})
            row = dict(object=obj['id'], true_young_pa=obj['young'], true_friction=obj['friction'],
                       fit_complete=bool(selected.get('success')))
            if row['fit_complete']:
                row.update(selected_young_pa=selected['scale'], selected_friction=selected['friction'],
                    absolute_relative_young_error=abs(selected['scale']/obj['young']-1),
                    absolute_friction_error=abs(selected['friction']-obj['friction']),
                    calibration_position_rmse_m=selected['calibration_rmse_m'],
                    scale_at_grid_edge=selected['scale_at_grid_edge'],
                    friction_at_grid_edge=selected['friction_at_grid_edge'])
            rows.append(row)
        valid = [r for r in rows if r['fit_complete']]
        splits[split] = dict(expected=len(rows), complete=len(valid), cases=rows,
            mean_absolute_relative_young_error=float(np.mean([r['absolute_relative_young_error'] for r in valid])) if valid else None,
            mean_absolute_friction_error=float(np.mean([r['absolute_friction_error'] for r in valid])) if valid else None)
    result = dict(scope='Post-fit parameter accuracy diagnostic. Truth labels used only for scoring. '
        'Finite-grid point estimates are not a posterior or calibrated uncertainty; fine/coarse model '
        'discrepancy can bias effective parameter estimates. Prediction accuracy is reported separately.',
        source_sha256=sources, auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), splits=splits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({s: {k: v for k, v in r.items() if k != 'cases'} for s, r in splits.items()}))


if __name__ == '__main__':
    main()
