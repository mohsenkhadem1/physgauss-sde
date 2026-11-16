"""Tabulate v02 calibration fits (no held-out data): per object and rung, posterior mean/sd of parameters, evidence,
calibration fit quality, tempering, and the M3 identifiability numbers.  usage: python -I v02_params.py
-> runs/pokeflex-v02/fit-summary.json (copied to reports/pokeflex-v02/)"""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_model import ROOT, OUT2, WORK2, OBJECTS


def main():
    out = {}
    for o in OBJECTS:
        f = json.loads((WORK2/'fit'/f'{o}.json').read_text())
        out[o] = {k: v for k, v in f.items() if k != 'rungs'}
        out[o]['rungs'] = {}
        for r, v in f['rungs'].items():
            p = v['posterior']
            out[o]['rungs'][r] = dict(log_evidence_selection=v['log_evidence_selection'], log_evidence_untempered=v['log_evidence_untempered'],
                                      BIC_neff=v['BIC_neff'], cal_force_rmse_best_N=v['cal_force_rmse_best_N'], cal_surface_rms_best_mm=v['cal_surface_rms_best_mm'],
                                      surface_clamped_mass=v.get('posterior_mass_surface_clamped'),
                                      params={k: (dict(mean=round(x['mean'], 4), sd=round(x['sd'], 4)) if isinstance(x, dict) and 'mean' in x else x)
                                              for k, x in p.items() if k not in ('r_marginal',)},
                                      r_marginal=p.get('r_marginal'))
            print(f"{o:18s} {r:4s} logZ {v['log_evidence_selection']:8.2f}  cal F rmse {v['cal_force_rmse_best_N']:6.2f}  S {v['cal_surface_rms_best_mm']:5.2f}  " +
                  '  '.join(f"{k} {x['mean']:.3g}±{x['sd']:.2g}" for k, x in p.items() if isinstance(x, dict) and 'mean' in x and k != 'nu')
                  + (f"  corr(kc,ks) {p['corr_log_kc_log_ks']:+.2f} r_sd/prior {p['r_sd_over_prior_sd']:.2f}" if r == 'M3' else ''))
    (OUT2/'fit-summary.json').write_text(json.dumps(out, indent=1))
    rep = ROOT/'reports/pokeflex-v02'; rep.mkdir(exist_ok=True); shutil.copyfile(OUT2/'fit-summary.json', rep/'fit-summary.json')


if __name__ == '__main__':
    main()
