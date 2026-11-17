"""Collect runs/patch-roles-v02/step3/cases/*/result-s*.json -> step3/summary.json, markdown table, risk calibration."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]/'runs/patch-roles-v02/step3'


def row(d):
    m = d['material']; fh = d.get('force_history', {})
    post = d['config_posterior']
    return dict(case=d['case'], seed=d['seed'], map_cfg=d['map_config'], post={k: round(v, 3) for k, v in post.items() if v > 1e-3},
                p_true=d['true_config_mass'], p_force_at_press=d.get('p_force_patch_at_press'), loc_err=d.get('force_loc_map_config_mean_err_mm'),
                map_force=d.get('map_force_centres_mm'), map_support=d.get('map_support'),
                force_cov=fh.get('cover90'), Fy=fh.get('final_Fy_mean'), Fy90=fh.get('final_Fy_90'), Fy_true=fh.get('final_Fy_true'),
                logE_err=m['logE_err'], logE_cov=m['logE_cover90'], p_xc=m['p_xc_true'], p_c8=m['p_contrast8'],
                D2=d['D2']['rmse_mm'], D2_sd=d['D2']['pred_sd_mm'], D2_cov=d['D2']['cover90'], D2_risk=d['D2']['risk'], D2_exc=d['D2']['exceeded'],
                H=d['D1_hidden']['rmse_mm'], H_sd=d['D1_hidden']['pred_sd_mm'], H_cov=d['D1_hidden']['cover90'], H_risk=d['D1_hidden']['risk'],
                H_exc=d['D1_hidden']['exceeded'], decision=d['design']['decision'],
                design=[(r['action'], round(r['expected_risk'], 3)) for r in d['design']['rows']], logZ=d['config_log_evidence'],
                smc_min=d['compute']['smc_seconds']/60)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ''          # '' or 'predall-'
    rows = [row(json.loads(p.read_text())) for p in sorted(R.glob(f'cases/*/result-{tag}s*.json')) if 'oracleK' not in str(p)]
    (R/f'summary{tag}.json').write_text(json.dumps(rows, indent=1, default=float))
    f = lambda v, n=2: '-' if v is None else f'{v:.{n}f}'
    print('| case | s | MAP (posterior) | P(true) | P(F@press) | loc mm | F cov | Fy end [90%] | dlogE | D2 rmse/sd (cov) | D2 risk/exc | D1 rmse/sd (cov) | D1 risk/exc | min |')
    print('|'+'---|'*14)
    for r in rows:
        fy = '-' if r['Fy'] is None else f"{r['Fy']:.1f} [{r['Fy90'][0]:.1f}, {r['Fy90'][1]:.1f}] / {r['Fy_true']:.0f}"
        print(f"| {r['case']} | {r['seed']} | {r['post']} | {r['p_true']:.3f} | {f(r['p_force_at_press'])} | {f(r['loc_err'], 1)} | {f(r['force_cov'])} | {fy} | "
              f"{r['logE_err']:.3f} | {r['D2']:.3f}/{r['D2_sd']:.3f} ({r['D2_cov']:.2f}) | {r['D2_risk']:.2f}/{'Y' if r['D2_exc'] else 'n'} | "
              f"{r['H']:.3f}/{r['H_sd']:.3f} ({r['H_cov']:.2f}) | {r['H_risk']:.2f}/{'Y' if r['H_exc'] else 'n'} | {r['smc_min']:.0f} |")
    cal = {}
    for t, rk, ek in (('D2', 'D2_risk', 'D2_exc'), ('D1', 'H_risk', 'H_exc')):
        for lo, hi in ((0, .1), (.1, .5), (.5, 1.01)):
            s = [r for r in rows if lo <= r[rk] < hi]
            cal[f'{t} [{lo},{hi})'] = dict(n=len(s), mean_risk=float(np.mean([r[rk] for r in s])) if s else None,
                                         exceed=float(np.mean([r[ek] for r in s])) if s else None)
    allr = [(r[k], r[e]) for r in rows for k, e in (('D2_risk', 'D2_exc'), ('H_risk', 'H_exc'))]
    cal['brier'] = float(np.mean([(p-float(x))**2 for p, x in allr])); cal['mean_risk'] = float(np.mean([p for p, _ in allr])); cal['exceed_rate'] = float(np.mean([x for _, x in allr]))
    print(json.dumps(cal, indent=1)); (R/f'calibration{tag}.json').write_text(json.dumps(cal, indent=1))


if __name__ == '__main__':
    main()
