"""Truth-free observability sweep: reveal more of the originally tip-loaded stem.

    This is posterior predictive design, not an additional held-out accuracy test.
"""
from pathlib import Path
import json
import numpy as np
from scripts.gap_first_v01.stem_inference import prior,expected_risk
from scripts.gap_first_v01.stem_mechanics import response_row


def main():
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'scripts/gap_first_v01/stem_protocol.json').read_text())
    L,F=.12,.02
    first=response_row(L,L,L,F)
    p=prior(c,257).update(first,.010,c['initial_sigma_m'])
    target=response_row(.75*L,.75*L,L,F)
    rows=[]
    for fraction in [1,.99,.98,.97,.95,.90,.80,.60,.40,.20,0]:
        row=response_row(fraction*L,L,L,F)
        action=dict(row=row,sigma=c['look_sigma_m'],cost=c['look_cost_in_failure_probability_units'])
        sv=np.linalg.svd(np.stack([first,row])/c['initial_sigma_m'],compute_uv=False)
        score=expected_risk(p,target,action,c['risk_tolerance_m'],321)
        rows.append(dict(lowest_visible_fraction=fraction,observation='original tip-force response at revealed marker',
            force_location_fraction=1,measurement_location_fraction=fraction,
            whitened_compliance_jacobian_singular_values=sv.tolist(),**score))
    result=dict(scope='Truth-free posterior predictive visibility sweep at y_tip=10mm. Not measured real-video calibration.',
        grid=257,predictive_bins=321,prior_risk=p.predictions(target,c['risk_tolerance_m'])['posterior_mean_risk'],rows=rows)
    (root/'reports/gap-first-v01/stem/visibility-sweep.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
