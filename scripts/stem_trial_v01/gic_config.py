"""Write a GIC config for a stem-trial episode. Usage: gic_config.py EPISODE OUT.json [init_logE init_nu]"""
import json, sys
from pathlib import Path
P = json.loads((Path(__file__).parent/'protocol.json').read_text())
ep, out = sys.argv[1], sys.argv[2]
init_E = float(sys.argv[3]) if len(sys.argv) > 3 else 5.6
init_nu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
dx = 0.08/6
cfg = {
 "data": {"xyz_min": [-0.3, 0.45, -0.2], "xyz_max": [0.3, 0.75, 0.2]},
 "gs": {"eval": True, "is_blender": True, "timenet": True, "test_iterations": [5000, 6000, 7000],
        "save_iterations": [7000, 10000, 20000, 30000, 40000], "quiet": False, "iteration": 40000,
        "data_device": "cuda", "load2gpu_on_the_fly": False},
 "physics": {"id": 0, "material": 10, "rho": 1000, "init_E": init_E, "init_nu": init_nu,
   "voxel_size": dx, "mpm_iter_cnt": 500, "n_frames": P['episodes'][ep]['frames'], "fps": P['episodes'][ep]['fps'],
   "gravity": [0, -9.8, 0], "bc": {"cylinder_clamp": [[-0.30, 0.6, 0.0], [-0.20, 0.6, 0.0], 0.07, 0]},
   "density_grid_size": 8*dx, "density_min_th": 0.5, "density_max_th": 0.8, "opacity_threshold": 0.005,
   "random_sample": False, "img_loss": True, "geo_loss": True, "w_img": 0.0, "w_alp": 1.0, "w_geo": 1.0,
   "params": {"Youngs modulus": {"lr_decay": True, "init_lr": 0.2, "final_lr": 0.01, "max_steps": 60},
              "Poisson ratio": {"lr_decay": True, "init_lr": 0.05, "final_lr": 0.0025, "max_steps": 60}},
   "iter_cnt": 100, "vel_iter_cnt": 100, "vel_estimation_frames": 4, "vel_lr": 0.025, "init_vel": [0.0, 0.0, 0.0]}}
cfg['notes'] = ('Native torus-config hyperparameters. Stem-specific: fps/frames, voxel_size = section/6, '
                'density_grid_size = 8 dx so reconstructed particle spacing equals the native particle volume '
                '(dx/2)^3, mpm_iter_cnt for CFL margin, native sticky cylinder collider as clamp, init_E one decade '
                'below the passive-equivalent value (torus config used one decade below).')
Path(out).write_text(json.dumps(cfg, indent=1))
