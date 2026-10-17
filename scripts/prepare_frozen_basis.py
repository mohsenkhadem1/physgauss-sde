"""Freeze the original recorded cube projection, without selecting from results."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
reference = json.loads((ROOT/"examples/neural-marginal/model_constants.json").read_text())
source = '''"""Original recorded cube projection; rank cut crosses a tied eigenspace.

Keeping a fixed projection is necessary for cross-backend reproducibility.
This is a declared rank-three covariance surrogate, not the full RBF kernel.
The constants predate calibration/evaluation of version 0.8.
"""
import torch

'''
for name, key in (("REFERENCE_REST", "rest_nodes_m"), ("REFERENCE_EDGES", "edges"), ("REFERENCE_BASIS", "internal_spatial_basis")):
    source += name+" = "+repr(reference[key])+"\n"
source += '''

def frozen_cube_basis(mesh, rank, length_scale):
    if rank != 3 or length_scale != .45 or tuple(mesh.rest.shape) != (8, 3) or tuple(mesh.edges.shape) != (19, 2):
        return None
    rest = mesh.rest.new_tensor(REFERENCE_REST)
    edges = torch.tensor(REFERENCE_EDGES, dtype=mesh.edges.dtype, device=mesh.edges.device)
    if not torch.equal(mesh.rest, rest) or not torch.equal(mesh.edges, edges):
        return None
    return mesh.rest.new_tensor(REFERENCE_BASIS)
'''
(ROOT/"src/physgauss/frozen_basis.py").write_text(source, encoding="utf-8")
plan = json.loads((ROOT/"configs/calibrated_noise_study.json").read_text())
plan["interval_test_seeds"] = list(range(83001, 83021))
plan["spatial_basis_policy"] = "frozen original cube rank-three projection; no backend-dependent tied-eigenspace selection"
(ROOT/"configs/calibrated_noise_frozen_basis_study.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
