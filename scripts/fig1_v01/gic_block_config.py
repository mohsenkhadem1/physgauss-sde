"""GIC config for the block scenario. Usage: gic_block_config.py OUT.json [init_logE init_nu]"""
import json, sys
from pathlib import Path
out = sys.argv[1]; init_E = float(sys.argv[2]) if len(sys.argv) > 2 else 4.5; init_nu = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
dx = 0.10/8
cfg = {"data": {"xyz_min": [-0.2, 0.45, -0.15], "xyz_max": [0.2, 0.75, 0.15]},
 "gs": {"eval": True, "is_blender": True, "timenet": True, "test_iterations": [5000, 6000, 7000],
        "save_iterations": [7000, 10000, 20000, 30000, 40000], "quiet": False, "iteration": 40000,
        "data_device": "cuda", "load2gpu_on_the_fly": False},
 "physics": {"id": 0, "material": 10, "rho": 1000, "init_E": init_E, "init_nu": init_nu, "voxel_size": dx, "mpm_iter_cnt": 300,
   "n_frames": 25, "fps": 48, "gravity": [0, -9.8, 0], "bc": {"ground": [[0, 0.53, 0], [0, 1, 0], 0]},
   "density_grid_size": 8*dx, "density_min_th": 0.5, "density_max_th": 0.8, "opacity_threshold": 0.005,
   "random_sample": False, "img_loss": True, "geo_loss": True, "w_img": 0.0, "w_alp": 1.0, "w_geo": 1.0,
   "params": {"Youngs modulus": {"lr_decay": True, "init_lr": 0.2, "final_lr": 0.01, "max_steps": 60},
              "Poisson ratio": {"lr_decay": True, "init_lr": 0.05, "final_lr": 0.0025, "max_steps": 60}},
   "iter_cnt": 100, "vel_iter_cnt": 100, "vel_estimation_frames": 4, "vel_lr": 0.025, "init_vel": [0.0, 0.0, 0.0]},
 "notes": "Native torus hyperparameters; native sticky ground plane at the glued bottom face; voxel = height/8; density 1000 (correct here)."}
Path(out).write_text(json.dumps(cfg, indent=1))
