"""Striker adapter check on GIC's simulator: free particle block, no gravity/colliders; total momentum
(particles + sphere) must be conserved and the sphere must rebound."""
import sys, os, json
from pathlib import Path
from argparse import Namespace
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'runs/gap-crossmethod-v02/vendor/gic')); sys.path.insert(0, str(Path(__file__).parent))
out = Path(sys.argv[1]); spec = dict(r=0.02, m=0.2, kappa=1e11, c0=[0, .6, .052], v0=[0, 0, -1.0], rho_config=1000.)
(out.parent/'striker-spec.json').write_text(json.dumps(spec)); os.environ['STEM_STRIKER_JSON'] = str(out.parent/'striker-spec.json')
import numpy as np, torch, taichi as ti
import striker_adapter as SA
SA.install()
ti.init(arch=ti.cuda, device_memory_fraction=.12)
from simulator import Simulator
from new_trajectory import gen_xyz_list
g = np.stack(np.meshgrid(np.arange(-.06, .06, .005), np.arange(.57, .63, .005), np.arange(-.03, .03, .005), indexing='ij'), -1).reshape(-1, 3)
vol = torch.tensor(g, dtype=torch.float32, device='cuda')
sim = Simulator(Namespace(gravity=[0, 0, 0], voxel_size=.01, fps=48, mpm_iter_cnt=400, vel=[0, 0, 0], rho=1000., bc={},
                          mat_params=dict(material=10, E=2e5, nu=.4)), vol)
seq = torch.stack(gen_xyz_list(sim, 6, diff=False)).cpu().numpy()
h = np.array(SA.STATE['history']); n = len(g); mp = 1000*(.005)**3
dt = 1/48/400; vbody = (seq[-1]-seq[-2]).mean(0)*48  # mean particle velocity over the last frame
p_body = n*mp*vbody[2]; p_s = spec['m']*h[-1][6]
print(json.dumps(dict(sphere_vz_final=float(h[-1][6]), body_momentum_z=float(p_body), total=float(p_body+p_s), initial=-0.2)))
