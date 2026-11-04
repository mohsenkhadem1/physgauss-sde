"""Persist the feature coordinates a posterior was trained in; reject unsafe resumes.

Use load_transform on resume, never rerun randomized PCA for saved weights.
Legacy bare state_dict checkpoints cannot establish this binding and must not be
silently upgraded. This module does not read observations or evaluation targets.
"""
import hashlib
import math
import numbers
import json
from pathlib import Path

import torch


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _digest_tensors(tensors):
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_transform(tensors):
    mean, basis, scale = (tensors[k] for k in ('mean', 'basis', 'scale'))
    if mean.ndim != 1 or basis.ndim != 2 or scale.ndim != 1:
        raise ValueError('Invalid PCA dimensions')
    if basis.shape != (mean.numel(), scale.numel()) or scale.numel() == 0:
        raise ValueError('Inconsistent PCA dimensions')
    if not all(torch.isfinite(t).all() for t in (mean, basis, scale)) or not (scale > 0).all():
        raise ValueError('Nonfinite PCA or nonpositive scale')


def save_transform(path, *, mean, basis, scale, bank_path, plan_path):
    """Save once, binding exact PCA coordinates to immutable bank and plan bytes."""
    if Path(path).exists():
        raise FileExistsError('Transform already frozen; load it instead')
    tensors = {k: v.detach().cpu().clone() for k, v in
               dict(mean=mean, basis=basis, scale=scale).items()}
    _validate_transform(tensors)
    payload = dict(schema=1, tensors=tensors, tensors_sha256=_digest_tensors(tensors),
                   bank_sha256=file_sha256(bank_path), plan_sha256=file_sha256(plan_path))
    _atomic_save(path, payload)


def load_transform(path, *, bank_path, plan_path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 1:
        raise ValueError('Unsupported transform schema')
    for key, source in [('bank_sha256', bank_path), ('plan_sha256', plan_path)]:
        if payload[key] != file_sha256(source):
            raise ValueError(f'Transform {key} mismatch')
    _validate_transform(payload['tensors'])
    if payload['tensors_sha256'] != _digest_tensors(payload['tensors']):
        raise ValueError('Transform tensor digest mismatch')
    return {k: t.to(device) for k, t in payload['tensors'].items()}


def save_bound_model(path, model, *, transform_path, bank_path, plan_path):
    # Validate the transform and its underlying bank/plan before binding weights.
    load_transform(transform_path, bank_path=bank_path, plan_path=plan_path)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _atomic_save(path, dict(schema=1, state_dict=state, state_sha256=_digest_tensors(state),
                           transform_sha256=file_sha256(transform_path),
                           bank_sha256=file_sha256(bank_path), plan_sha256=file_sha256(plan_path)))


def load_bound_model(path, model, *, transform_path, bank_path, plan_path):
    """Reject bare legacy checkpoints, changed bank/plan/PCA, or damaged weights."""
    load_transform(transform_path, bank_path=bank_path, plan_path=plan_path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 1 or 'state_dict' not in payload:
        raise ValueError('Unbound legacy checkpoint; retrain or recover original transform')
    for key, source in [('transform_sha256', transform_path), ('bank_sha256', bank_path),
                        ('plan_sha256', plan_path)]:
        if payload[key] != file_sha256(source):
            raise ValueError(f'Model {key} mismatch')
    if payload['state_sha256'] != _digest_tensors(payload['state_dict']):
        raise ValueError('Model state digest mismatch')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model


def inference_seal(*, inference_path, model_paths, transform_path, bank_path, plan_path):
    """Return hashes for a seal written before forecast/evaluation; not a signature."""
    load_transform(transform_path, bank_path=bank_path, plan_path=plan_path)
    for model_path in model_paths.values():
        payload = torch.load(model_path, map_location='cpu', weights_only=True)
        if payload.get('schema') != 1 or 'state_dict' not in payload:
            raise ValueError('Cannot seal unbound legacy model')
        for key, source in [('transform_sha256', transform_path), ('bank_sha256', bank_path),
                            ('plan_sha256', plan_path)]:
            if payload[key] != file_sha256(source):
                raise ValueError(f'Cannot seal model: {key} mismatch')
        if payload['state_sha256'] != _digest_tensors(payload['state_dict']):
            raise ValueError('Cannot seal model: state digest mismatch')
    return dict(schema=1, inference_sha256=file_sha256(inference_path),
                models_sha256={k: file_sha256(v) for k, v in sorted(model_paths.items())},
                transform_sha256=file_sha256(transform_path), bank_sha256=file_sha256(bank_path),
                plan_sha256=file_sha256(plan_path), forecast_target_access=False)


def relative_observation_noise(batch_size, frames, points, sigma, *, mask=None,
                               generator=None, device='cpu', dtype=torch.float32):
    """Independent absolute-coordinate errors, then shared frame-0 subtraction.

    Returns flattened [batch, frames*points*3], optionally masked AFTER subtraction.
    For each tracked coordinate: Var(delta_t)=2*sigma**2 for t>0,
    Cov(delta_t,delta_s)=sigma**2 for distinct nonzero t,s, and delta_0=0.
    Assumes independent per-frame absolute errors and persistent point identities.
    A known noise-free initial reference would require a different noise model.
    """
    dimensions=(batch_size,frames,points)
    if any(isinstance(n,bool) or not isinstance(n,numbers.Integral) or n<1 for n in dimensions):
        raise ValueError('Positive integer dimensions required')
    if not isinstance(sigma,numbers.Real) or not math.isfinite(float(sigma)) or sigma<0:
        raise ValueError('Finite nonnegative sigma required')
    errors = sigma * torch.randn((batch_size, frames, points, 3), generator=generator,
                                 device=device, dtype=dtype)
    relative = (errors - errors[:, :1]).reshape(batch_size, -1)
    if mask is not None:
        mask = torch.as_tensor(mask, device=device)
        if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != frames * points * 3:
            raise ValueError('Mask must be a boolean flattened coordinate mask')
        relative = relative[:, mask]
    return relative
