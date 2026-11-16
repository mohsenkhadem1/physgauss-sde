"""Collect runs/patch-roles-v01/*/result-s*.json into summary.json and markdown tables."""
import json, sys
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[2] / 'runs/patch-roles-v01'


def fmt(x, f='{:.3f}'):
    if x is None: return '-'
    if isinstance(x, list): return '/'.join(fmt(v, f) for v in x) if x else '-'
    if isinstance(x, bool): return 'yes' if x else 'no'
    return f.format(x)


def main():
    rows, des = [], []
    for p in sorted(RUNS.glob('*/result-s*.json')):
        r = json.loads(p.read_text())
        for st, s in r['stages'].items():
            mt = s.get('matched_true_config', {}); g = s['global']; pr = s['prediction']
            cmds = pr['commands']
            rows.append(dict(scene=p.parent.name, seed=r['seed'], stage=st, true=s['true_config'], map=s['map_config'],
                             mass=s['true_config_mass'], pF=s['p_force_within_15mm_of_truth'], pS=s['p_support_within_15mm_of_truth'],
                             spur=s['expected_spurious_patches'], locF=mt.get('force_location_error_mm'), locS=mt.get('support_location_error_mm'),
                             dlogE=g['logE0_error'], Ecov=g['logE0_cover90'], dlogk=mt.get('support_logk_error'), kcov=mt.get('support_logk_cover90'),
                             fcov=mt.get('force_cover90'), rcov=g['resultant_force_cover90'], tau=g.get('tau_mean'), tauI=g.get('tau_interval90'),
                             corr=g.get('corr_logE0_logtau'),
                             hid=pr['hidden_rmse_mm'], hidsd=pr['hidden_pred_sd_mm'], hidcov=pr['hidden_cover90'],
                             cmd=float(np.mean([c['rmse_mm'] for c in cmds])), cmdsd=float(np.mean([c['pred_sd_mm'] for c in cmds])),
                             cmdcov=float(np.mean([c['cover90'] for c in cmds])),
                             risk=[c['risk'] for c in cmds], exc=[c['exceeded'] for c in cmds],
                             sec=s['compute']['smc_seconds'], post=s['config_posterior'], logZ=s['config_log_evidence']))
            des.append(dict(scene=p.parent.name, seed=r['seed'], stage=st, decision=s['design']['decision'],
                            rows=s['design']['rows'], executed=r.get('executed_probe')))
        if 'regression_published_1F1S' in r: rows[-1]['regression'] = r['regression_published_1F1S']
    (RUNS / 'summary.json').write_text(json.dumps(dict(rows=rows, design=des), indent=1, default=float))
    print('| scene | seed | stage | true | MAP | P(true cfg) | P(F near true) | P(S near true) | E[#spurious] | loc err F / S (mm) | |dlogE0| (90%) | |dlogk| (90%) | force cov90 | resultant cov90 | hidden RMSE/sd mm (cov90) | new-cmd RMSE/sd mm (cov90) | SMC s |')
    print('|' + '---|' * 17)
    for r in rows:
        print(f"| {r['scene']} | {r['seed']} | {r['stage']} | {r['true']} | {r['map']} | {r['mass']:.3f} | {fmt(r['pF'],'{:.2f}')} | {fmt(r['pS'],'{:.2f}')} | {r['spur']:.2f} | "
              f"{fmt(r['locF'],'{:.0f}')} / {fmt(r['locS'],'{:.0f}')} | {r['dlogE']:.2f} ({fmt(r['Ecov'])}) | {fmt(r['dlogk'],'{:.2f}')} ({fmt(r['kcov'])}) | "
              f"{fmt(r['fcov'],'{:.2f}')} | {r['rcov']:.2f} | {r['hid']:.4f}/{r['hidsd']:.4f} ({r['hidcov']:.2f}) | {r['cmd']:.4f}/{r['cmdsd']:.4f} ({r['cmdcov']:.2f}) | {r['sec']:.0f} |")
    print()
    for r in rows:
        print(r['scene'], r['seed'], r['stage'], 'post', {k: round(v, 3) for k, v in r['post'].items() if v > 1e-3},
              'logZ', {k: round(v, 1) for k, v in r['logZ'].items()}, 'tau', r['tau'], r['tauI'], 'corr', r['corr'], 'risk', fmt(r['risk'], '{:.2f}'), 'exceeded', r['exc'])
        if 'regression' in r: print('REGRESSION', json.dumps(r['regression']))
    print()
    for d in des:
        print(d['scene'], d['seed'], d['stage'], 'decision:', d['decision'], 'executed:', d['executed'] and d['executed']['name'])
        for x in d['rows']:
            print(f"    {x['action']:24s} J={x['J']:.3f} E[r_b]={x['expected_risk_b']:.3f} E[hidden sd]={x['expected_hidden_sd_mm']*1000:.2f}um")


if __name__ == '__main__':
    main()
