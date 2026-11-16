"""Collect runs/patch-roles-v02/cases/*/result-s*.json into summary.json and a markdown table on stdout."""
import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]/'runs/patch-roles-v02'


def row(d):
    m = d['material']; fh = d.get('force_history', {})
    return dict(case=d['case'], seed=d['seed'], map_cfg=d['map_config'], p_true_cfg=d['true_config_mass'],
                p_force_at_press=d.get('p_force_patch_at_press'), E_nF=d.get('expected_n_force'), E_nS=d.get('expected_n_support'),
                loc_err_map_mm=d.get('force_loc_map_config_mean_err_mm'), map_support=d.get('map_support'),
                force_cov=fh.get('cover90'), force_cov_y=fh.get('cover90_y'), force_rmse_N=fh.get('rmse_N'),
                Fy_end=fh.get('final_Fy_mean'), Fy_end_90=fh.get('final_Fy_90'), Fy_true=fh.get('final_Fy_true'),
                logE_err=m['logE_err'], logE_cov=m['logE_cover90'], p_xc=m['p_xc_true'], p_c8=m['p_contrast8'], p_nu=m['p_nu_true'],
                sw_mm=m['sigma_w_mm_mean'], mat_map=m['map'],
                D2_rmse=d['D2']['rmse_mm'], D2_sd=d['D2']['pred_sd_mm'], D2_cov=d['D2']['cover90'], D2_z=d['D2']['z_rms'],
                D2_risk=d['D2']['risk'], D2_delta=d['D2']['delta_mm'], D2_exc=d['D2']['exceeded'],
                H_rmse=d['D1_hidden']['rmse_mm'], H_sd=d['D1_hidden']['pred_sd_mm'], H_cov=d['D1_hidden']['cover90'], H_z=d['D1_hidden']['z_rms'],
                H_risk=d['D1_hidden']['risk'], H_exc=d['D1_hidden']['exceeded'],
                decision=d['design']['decision'], best_probe=(d['design']['best_probe'] or {}).get('action'),
                best_probe_risk=(d['design']['best_probe'] or {}).get('expected_risk'), proceed_risk=d['design']['rows'][0]['expected_risk'],
                near=d['design'].get('near_map_force_candidates'),
                smc_min=d['compute']['smc_seconds']/60, logZ=d['config_log_evidence'], post=d['config_posterior'])


def main():
    rows = [row(json.loads(p.read_text())) for p in sorted(R.glob('cases/*/result-s*.json'))]
    (R/'summary.json').write_text(json.dumps(rows, indent=1, default=float))
    f = lambda v, n=2: '-' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))
    print('| case | s | MAP cfg | P(1F0S) | P(F@press) | loc err mm | F cov | Fy(0.5s) [90%] | dlogE (cov) | P(xc) | P(c=8) | D2 rmse/sd mm (cov) | D2 risk/exc | hidden rmse/sd (cov) | min |')
    print('|'+'---|'*15)
    for r in rows:
        fy = '-' if r['Fy_end'] is None else f"{r['Fy_end']:.1f} [{r['Fy_end_90'][0]:.1f},{r['Fy_end_90'][1]:.1f}]"
        print(f"| {r['case']} | {r['seed']} | {r['map_cfg']} | {f(r['p_true_cfg'],3)} | {f(r['p_force_at_press'])} | {f(r['loc_err_map_mm'],1)} | {f(r['force_cov'])} | {fy} | "
              f"{r['logE_err']:.3f} ({'y' if r['logE_cov'] else 'n'}) | {r['p_xc']:.2f} | {r['p_c8']:.2f} | {r['D2_rmse']:.3f}/{r['D2_sd']:.3f} ({r['D2_cov']:.2f}) | "
              f"{r['D2_risk']:.2f}/{'Y' if r['D2_exc'] else 'n'} | {r['H_rmse']:.3f}/{r['H_sd']:.3f} ({r['H_cov']:.2f}) | {r['smc_min']:.0f} |")


if __name__ == '__main__':
    main()


def calibration():
    """Threshold-risk calibration over all results (D2 and hidden-D1 targets), and coverage pooled by variant."""
    rows = json.loads((R/'summary.json').read_text()); out = {}
    for tgt, rk, ek in (('D2', 'D2_risk', 'D2_exc'), ('D1_hidden', 'H_risk', 'H_exc')):
        for lo, hi in ((0, .1), (.1, .5), (.5, 1.01)):
            s = [r for r in rows if lo <= r[rk] < hi]
            out[f'{tgt} risk [{lo},{hi})'] = dict(n=len(s), mean_risk=float(np.mean([r[rk] for r in s])) if s else None,
                                                 exceed_rate=float(np.mean([r[ek] for r in s])) if s else None)
    print(json.dumps(out, indent=1)); return out


if __name__ == '__main__' and len(__import__('sys').argv) > 1:
    calibration()
