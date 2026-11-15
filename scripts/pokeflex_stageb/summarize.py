"""Collect stage-B fit summaries for the report (no outcomes read): fit parameters, ranking scalars, calibration-fit
quality (calib_quality.json, calibration window only), PAC-NeRF rest-shape quality, coverage, compute.
usage: python -I summarize.py  -> prints markdown tables, writes runs/pokeflex-v01/stageB/summary.json"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sb_common import WORK, OUT, PRED_B, OBJECTS, obj_info

cq = json.loads((OUT/'stageB/calib_quality.json').read_text()) if (OUT/'stageB/calib_quality.json').exists() else {}

rows = {}
print('| object | PhysTwin params | PT dF/dd N/m | PT fit min | PAC E kPa | PAC nu | PAC dF/dd N/m | PAC rest chamfer mm (cal) | PAC held-out ICP res. mm (median) |')
print('|---|---|---|---|---|---|---|---|---|')
for obj in OBJECTS:
    r = {}
    f = WORK/'phystwin'/obj/'fit.json'
    if f.exists():
        z = json.loads(f.read_text()); r['PhysTwin'] = dict(params=z['params'], dFdd=z['ranking_dF_dd_N_per_m'], minutes=z['seconds']/60,
                                                            evals=z['n_evals'], loss=z['loss'])
    f = WORK/'pacnerf'/obj/'fit.json'
    if f.exists():
        z = json.loads(f.read_text())
        icp = [json.loads(p.read_text())['rest_shape'].get('icp', {}).get('residual_mm') for p in (PRED_B/'PAC-NeRF'/obj).glob('*-p*.json')]
        icp = [v for v in icp if v is not None]
        r['PAC-NeRF'] = dict(E=z['E'], nu=z['nu'], dFdd=z['ranking_dF_dd_N_per_m'], minutes=z['seconds']/60, loss=z['loss'],
                             rest_shape=z['rest_shape'], icp_median_mm=float(np.median(icp)) if icp else None,
                             icp_max_mm=float(np.max(icp)) if icp else None)
    for meth in ('PhysTwin', 'PAC-NeRF'):
        n = len(list((PRED_B/meth/obj).glob('*-p*.npz'))) if (PRED_B/meth/obj).exists() else 0
        r.setdefault(meth, {})['n_pred'] = n
    r['n_heldout'] = len(obj_info(obj)['heldout']); r['calib'] = cq.get(obj)
    rows[obj] = r
    pt, pa = r.get('PhysTwin', {}), r.get('PAC-NeRF', {})
    prm = pt.get('params'); ps = (f"Y={10**prm['logY']:.2f} N, r={prm['r_fac']:.2f}h, nn={prm['max_nn']}, el={prm['collide_elas']:.2f}, "
                                  f"fr={prm['collide_fric']:.2f}, dp={prm['dashpot']:.0f}, dr={prm['drag']:.1f}") if prm else '-'
    fmt = lambda v, s='{:.0f}': (s.format(v) if v is not None else '-')
    print(f"| {obj} | {ps} | {fmt(pt.get('dFdd'))} | {fmt(pt.get('minutes'))} | {fmt(pa.get('E') and pa['E']/1e3, '{:.1f}')} | "
          f"{fmt(pa.get('nu'), '{:.1f}')} | {fmt(pa.get('dFdd'))} | {fmt(pa.get('rest_shape', {}).get('chamfer_mm'), '{:.1f}')} | "
          f"{fmt(pa.get('icp_median_mm'), '{:.1f}')} |")
print()
print('| object | method | calib force RMSE N | along -a RMSE N | peak pred/meas N | chamfer_excl_tool mm | chamfer mm |')
print('|---|---|---|---|---|---|---|')
for obj in OBJECTS:
    c = rows[obj]['calib']
    if not c:
        continue
    for meth in ('PhysTwin', 'PAC-NeRF', 'null'):
        if meth in c:
            d = c[meth]
            print(f"| {obj} | {meth} | {d['force_rmse_N']:.2f} | {d['force_along_rmse_N']:.2f} | {d['peak_pred_N']:.1f}/{d['peak_meas_N']:.1f} | "
                  f"{d['chamfer_excl_tool_mm']:.2f} | {d['chamfer_mm']:.2f} |")
print()
print('ranking scalars (N/m): object, PhysTwin, PAC-NeRF (catalogue stiffness not read: reserved for the ranking check)')
for obj in OBJECTS:
    print(obj, rows[obj].get('PhysTwin', {}).get('dFdd'), rows[obj].get('PAC-NeRF', {}).get('dFdd'))
(OUT/'stageB/summary.json').write_text(json.dumps(rows, indent=1, default=float))
