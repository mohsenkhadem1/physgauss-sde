import pytest
import torch
from scripts.gaussian_boundary_flow_v02.checkpoint_contract import (
    save_transform, load_transform, save_bound_model, load_bound_model,
    inference_seal, relative_observation_noise, file_sha256)


def setup_contract(tmp_path):
    bank, plan, transform = [tmp_path / n for n in ('bank.npz', 'plan.json', 'pca.pt')]
    bank.write_bytes(b'bank'); plan.write_text('{}')
    args = dict(bank_path=bank, plan_path=plan)
    save_transform(transform, mean=torch.zeros(3), basis=torch.eye(3), scale=torch.ones(3), **args)
    return transform, args


def test_exact_roundtrip_and_seal(tmp_path):
    transform, args = setup_contract(tmp_path)
    loaded = load_transform(transform, **args)
    assert torch.equal(loaded['basis'], torch.eye(3))
    model = torch.nn.Linear(3, 2); restored = torch.nn.Linear(3, 2)
    path = tmp_path / 'model.pt'
    save_bound_model(path, model, transform_path=transform, **args)
    load_bound_model(path, restored, transform_path=transform, **args)
    x = torch.randn(4, 3)
    assert torch.equal(model(x), restored(x))
    inference = tmp_path / 'inference.npz'; inference.write_bytes(b'samples')
    seal = inference_seal(inference_path=inference, model_paths={'model': path},
                          transform_path=transform, **args)
    assert seal['models_sha256']['model'] == file_sha256(path)
    assert seal['inference_sha256'] == file_sha256(inference)
    with pytest.raises(FileExistsError):
        save_transform(transform, **loaded, **args)


@pytest.mark.parametrize('changed', ['bank_path', 'plan_path'])
def test_reject_changed_inputs(tmp_path, changed):
    transform, args = setup_contract(tmp_path)
    args[changed].write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        load_transform(transform, **args)


def test_reject_different_pca_and_legacy(tmp_path):
    transform, args = setup_contract(tmp_path)
    model = torch.nn.Linear(3, 2); path = tmp_path / 'model.pt'
    save_bound_model(path, model, transform_path=transform, **args)
    alternate = tmp_path / 'pca2.pt'
    save_transform(alternate, mean=torch.ones(3), basis=-torch.eye(3), scale=torch.ones(3), **args)
    with pytest.raises(ValueError, match='transform_sha256 mismatch'):
        load_bound_model(path, model, transform_path=alternate, **args)
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match='Unbound legacy'):
        load_bound_model(path, model, transform_path=transform, **args)


def test_reject_tampered_weights(tmp_path):
    transform, args = setup_contract(tmp_path)
    model = torch.nn.Linear(3, 2); path = tmp_path / 'model.pt'
    save_bound_model(path, model, transform_path=transform, **args)
    payload = torch.load(path, weights_only=True); payload['state_dict']['bias'].add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match='state digest'):
        load_bound_model(path, model, transform_path=transform, **args)


def test_relative_noise_covariance_and_mask():
    def draw(mask=None):
        return relative_observation_noise(100000, 3, 1, .002, mask=mask,
                                         generator=torch.Generator().manual_seed(442))
    noise = draw(); assert torch.count_nonzero(noise[:, :3]) == 0
    covariance = torch.cov(noise[:, [3, 6]].T)
    assert torch.allclose(covariance, torch.tensor([[8e-6, 4e-6], [4e-6, 8e-6]]), rtol=.025, atol=0)
    mask = torch.tensor([False, True, False, True, False, False, True, False, False])
    assert torch.equal(draw(mask), noise[:, mask])
    assert torch.equal(draw(), noise)
