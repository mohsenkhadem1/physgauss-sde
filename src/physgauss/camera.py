"""Known pinhole camera paths for the controlled benchmark."""
import math
import torch
from torch import Tensor


def orbit(times: Tensor, degrees=100.0, offset_degrees=0.0):
    if len(times) < 2 or float(times[-1] - times[0]) <= 0:
        raise ValueError("An orbit needs increasing times spanning a nonzero duration.")
    phase = (times - times[0]) / (times[-1] - times[0]) - 0.5
    angle = phase * math.radians(degrees) + math.radians(offset_degrees + 25)
    eye = torch.stack((2.4 * angle.sin(), torch.ones_like(angle) * 1.35,
                       2.4 * angle.cos()), dim=-1)
    target = eye.new_tensor([0.0, 0.51, 0.0])
    forward = torch.nn.functional.normalize(target - eye, dim=-1)
    up = torch.zeros_like(forward)
    up[:, 1] = 1
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=-1)
    camera_up = torch.linalg.cross(right, forward)
    # Camera rows: right, down, forward. Image y grows downwards.
    rotation = torch.stack((right, -camera_up, forward), dim=-2)
    return eye, rotation


def project(points: Tensor, eye: Tensor, rotation: Tensor, size: int):
    """Points [B,T,N,3], cameras [T,...]; return pixels, depth, Jacobian."""
    xyz = torch.einsum("tij,btnj->btni", rotation, points - eye[None, :, None])
    depth = xyz[..., 2]
    if bool((depth <= 0).any()):
        raise ValueError("A point moved behind the pinhole camera.")
    focal = size * 1.65
    mean = focal * xyz[..., :2] / depth[..., None] + (size - 1) / 2
    jac = xyz.new_zeros((*xyz.shape[:-1], 2, 3))
    jac[..., 0, 0] = focal / depth
    jac[..., 1, 1] = focal / depth
    jac[..., 0, 2] = -focal * xyz[..., 0] / depth.square()
    jac[..., 1, 2] = -focal * xyz[..., 1] / depth.square()
    return mean, depth, jac
