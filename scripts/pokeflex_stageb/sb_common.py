"""Stage-B shared helpers (no PokeFlex loader here: GPU-side adapters only read the exported inputs).

Inputs (exported by prep_inputs.py with the gated loader, free inputs + calibration window only):
  runs/pokeflex-v01/stageB/inputs/<Obj>/object.json
  runs/pokeflex-v01/stageB/inputs/<Obj>/cal_<take>-p<k>.npz   calibration poke: frames, T (K,4,4) T_WT, dep, F (offset-corrected
                                                              measured force, N), rest V/F, per-frame surface samples S (K,n,3)
                                                              and their tool-exclusion mask Skeep (K,n)
  runs/pokeflex-v01/stageB/inputs/<Obj>/ho_<take>-p<k>.npz    held-out poke: frames, T, dep, rest V/F (NO outcomes)
Predictions: runs/pokeflex-v01/pred-stageB/<method>/<Obj>/<take>-p<k>.{npz,json} in the format of
scripts/pokeflex_v01/pfx_common.save_pred (frames, force, surf_V, surf_F [, risk]) + <method>/<Obj>/fit.json."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2]
OUT = R/'runs/pokeflex-v01'
INP = OUT/'stageB/inputs'
WORK = OUT/'stageB/work'
PRED_B = OUT/'pred-stageB'
OBJECTS = json.loads((R/'scripts/pokeflex_v01/protocol.json').read_text())['objects_considered']
R_TOOL, L_TOOL, G_ACC = 0.010, 0.192, 9.81   # same tool as ours (realfem.py): flat-ended cylinder, end face at T_WT origin


def save_pred(method, obj, take, k, frames, force, meta, surf_V=None, surf_F=None, risk=None):
    """Same arrays/format as pfx_common.save_pred, rooted at pred-stageB."""
    d = PRED_B/method/obj; d.mkdir(parents=True, exist_ok=True)
    arr = dict(frames=np.asarray(frames, int), force=np.asarray(force, float))
    if surf_V is not None:
        arr.update(surf_V=np.asarray(surf_V, np.float32), surf_F=np.asarray(surf_F, np.int32))
    if risk is not None:
        arr['risk'] = np.array(float(risk))
    np.savez_compressed(d/f'{take}-p{k}.npz', **arr)
    (d/f'{take}-p{k}.json').write_text(json.dumps(meta, indent=1))


def obj_info(obj):
    return json.loads((INP/obj/'object.json').read_text())


def load_poke(obj, kind, take, k):
    return dict(np.load(INP/obj/f'{kind}_{take}-p{k}.npz'))


def tool_pose(T):
    a = T[..., :3, 2]/np.linalg.norm(T[..., :3, 2], axis=-1, keepdims=True)
    return T[..., :3, 3], a


def tool_keep(X, tip, axis, excl_radius=0.03):
    """Protocol's tool-exclusion rule (pokeflex.surface_error_mm): drop points within excl_radius of the tool axis from
    1 cm below the tip upwards."""
    r = X-tip; ax = r @ axis
    return ~((np.linalg.norm(r-np.outer(ax, axis), axis=1) < excl_radius) & (ax < 0.01))


def chamfer_one_way_mm(A, B):
    """mean distance from points A to nearest point of B (mm)."""
    from scipy.spatial import cKDTree
    return float(cKDTree(B).query(A)[0].mean()*1e3)


def loading_slope(fa, dep):
    """Stiffness-ranking scalar: least-squares slope of the predicted push force along -approach vs contact depth over
    the loading half (frames up to the deepest one, depth > 0), as ours' dF/dd."""
    j = int(np.argmax(dep)); d = np.maximum(dep[:j+1], 0); f = fa[:j+1]
    return float((f @ d)/(d @ d)) if (d @ d) > 0 else float('nan')
