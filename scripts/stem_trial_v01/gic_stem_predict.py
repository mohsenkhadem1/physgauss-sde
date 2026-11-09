"""Native GIC Simulator forecast of D2 from a fitted run, with the labelled force adapter.
Usage (cwd = GIC vendor, STEM_FORCE_JSON = D2 spec): gic_stem_predict.py MODEL_DIR OUT.npz"""
import sys, json
from pathlib import Path
from argparse import Namespace
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'runs/gap-crossmethod-v02/vendor/gic')); sys.path.insert(0, str(Path(__file__).parent))
import gic_force_adapter as A
assert A.install() is not None
import numpy as np, torch, taichi as ti
from simulator import Simulator
from new_trajectory import load_pcd_file, gen_xyz_list
P = json.loads((Path(__file__).parent/'protocol.json').read_text())
STATIONS = [(-.11, -.09), (-.01, .01), (.09, .11), (.18, .2001)]
model, out = Path(sys.argv[1]), Path(sys.argv[2])
phys = Namespace(**json.loads((model/'0-pred.json').read_text()))
ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=float(__import__('os').environ.get('STEM_TI_MEM', '0.3')))
vol = load_pcd_file(str(model), 40000)
sim = Simulator(phys, vol)
n = P['episodes']['D2']['frames']
seq = gen_xyz_list(sim, n, diff=False)
assert sim.succeed() and all(torch.isfinite(x).all() for x in seq), 'CFL/nonfinite failure (retained)'
xyz = torch.stack(seq).cpu().numpy(); ref = vol.cpu().numpy()
cent = np.stack([[xyz[f][(ref[:, 0] >= lo) & (ref[:, 0] < hi)].mean(0) for lo, hi in STATIONS] for f in range(n)])
counts = [int(((ref[:, 0] >= lo) & (ref[:, 0] < hi)).sum()) for lo, hi in STATIONS]
np.savez_compressed(out, xyz=xyz.astype(np.float32), ref=ref, station_centroid=cent, station_counts=np.array(counts),
                    params=json.dumps(vars(phys)), adapter_audit=json.dumps(A.AUDIT))
print(dict(out=str(out), counts=counts, audit=A.AUDIT, final_disp_mm=((cent[-1]-cent[0])*1e3).round(2).tolist()))
