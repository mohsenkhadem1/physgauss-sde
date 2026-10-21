"""Fit a direct neural teacher with prepartitioned TRAIN calibrated controls.

Independent diagonal controls, calibration responses, VAL controls and TEST
labels are never opened. Passive VAL remains a development diagnostic. The
primary controlled teacher is the fixed-step final checkpoint, not a checkpoint
selected using the later control assessments.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from train_gaussian_forward import collect_records, evaluate, make_model, sha
from gaussian_probe_roles import load_role_manifest, require_teacher_control, teacher_control_label

ROLE_SHA256 = '6d2781a71a24ebfa9cb0691ce0dd7ae889c15eae1b38ca3164618523d5552c84'


def collect_control_records(dataset, pool, role_path, *, require_complete=True):
    dataset, pool = Path(dataset), Path(pool)
    roles = load_role_manifest(role_path, expected_sha256=ROLE_SHA256)
    if sha(dataset / 'plan.json') != roles['dataset_plan_sha256']:
        raise ValueError('Control role manifest refers to a different source plan')
    plan = json.loads((dataset / 'plan.json').read_text())
    cases = {c['episode_id']: c for c in plan['episodes']}
    rows, missing = [], []
    for name in roles['teacher_fit_ids']:
        case = cases[name]
        # Establish eligibility before opening ANY private control labels.
        for action_id in roles['eligible_control_ids']:
            require_teacher_control(roles, name, action_id, case['split'])
        source = dataset / 'episodes' / name
        private = pool / 'private' / name
        validation = private / 'validation.json'
        if not validation.exists():
            missing.append(name)
            continue
        if not json.loads(validation.read_text()).get('passed'):
            raise ValueError(f'Failed control capture audit: {name}')
        recipe = json.loads((private / 'truth/recipe.json').read_text())
        if recipe != case:
            raise ValueError('Control recipe differs from passive TRAIN recipe')
        if sha(private / 'provenance/isaac_calibrated_probes_v02.py') != roles['generation_source_sha256']:
            raise ValueError('Controlled generation source changed')
        summary = json.loads((private / 'summary.json').read_text())
        if not summary.get('completed') or not summary.get('source_sha256'):
            raise ValueError('Control generation is not complete with source provenance')
        for source_name, expected in summary['source_sha256'].items():
            if Path(source_name).name != source_name or sha(private / 'provenance' / source_name) != expected:
                raise ValueError('Control generation provenance differs from summary')
        menu = json.loads((pool / 'public' / name / 'action-menu.json').read_text())
        actions = {a['action_id']: a for a in menu['candidates']}
        with np.load(source / 'reference.npz') as a:
            rest = a['rest_points_m'].astype(np.float32)
            tets = a['tetrahedra']
        spec = json.loads((private / 'truth/boundaries.json').read_text())
        for action_id in roles['eligible_control_ids']:
            action = actions[action_id]
            declared = roles['controls'][action_id]
            if (action['force_type_index'] != declared['force_type_index'] or
                    not np.array_equal(action['peak_force_N'], declared['peak_force_N'])):
                raise ValueError('Actual action differs from frozen control menu')
            physical, bc, force = teacher_control_label(roles, case, spec, rest, action)
            if not np.allclose(physical[10:16], np.array(menu['known_handle_centers_m']).ravel(), atol=1e-7):
                raise ValueError('Known control handle geometry differs')
            path = private / action_id
            control = json.loads((path / 'control.json').read_text())
            if control != action:
                raise ValueError('Saved control differs from public action')
            with np.load(path / 'trajectory.npz') as a:
                q = a['positions_m'].astype(np.float32)
                times = a['times_s'].astype(np.float32)
            if q.shape != (31, len(rest), 3) or not np.allclose(times, np.arange(31) / 10, atol=1e-6):
                raise ValueError('Expected actual 10Hz saved control grid; no interpolation')
            certificate = json.loads((path / 'capture-audit.json').read_text())
            if len(certificate) != len(q) or any(not c.get('passed') or
                    hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() != c['positions_sha256']
                    for frame, c in zip(q, certificate)):
                raise ValueError('Control positions differ from audited capture certificates')
            rows.append(dict(path=str(path), split='train', episode_id=name + '/' + action_id,
                group_id=str(dataset) + ':' + case['group_id'], physical=physical, bc=bc, force=force,
                rest=rest, tets=tets, q=q, times=times, scale_m=max(float(np.sqrt(np.mean((q-rest)**2))), .0004),
                trajectory_sha256=sha(path / 'trajectory.npz'), recipe_sha256=case['recipe_sha256'],
                role='teacher_fit_control', action_id=action_id))
    if require_complete and missing:
        raise ValueError(f'Frozen teacher controls incomplete: {missing}')
    if not rows:
        raise ValueError('No eligible completed TRAIN controls')
    return rows


def decimate_passive_training(records, times):
    """Use real saved passive frames on the same grid; keep full VAL grid."""
    result = []
    for record in records:
        row = dict(record, role='passive')
        if row['split'] == 'train':
            ids = np.array([int(np.argmin(abs(row['times'] - t))) for t in times])
            if len(set(ids)) != len(times) or not np.allclose(row['times'][ids], times, atol=1e-6):
                raise ValueError('Passive data lack actual requested saved timestamps')
            row['q'], row['times'] = row['q'][ids], row['times'][ids]
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--roles', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--points', type=int, default=160)
    p.add_argument('--times', type=int, default=12)
    p.add_argument('--probe-fraction', type=float, default=.5)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=20261113)
    a = p.parse_args()
    if not 0 < a.probe_fraction < 1:
        p.error('Both passive and control training require positive probability')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    frozen = json.loads(a.catalog.read_text())
    keys = {(str(Path(r['path']).resolve()), r['recipe_sha256']) for r in frozen}
    records = [r for r in collect_records(a.roots) if (str(Path(r['path']).resolve()), r['recipe_sha256']) in keys]
    if len(records) != len(keys):
        raise ValueError('Frozen passive catalog unavailable or changed')
    controls = collect_control_records(a.dataset, a.pool, a.roles)
    records = decimate_passive_training(records, controls[0]['times'])
    passive = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'validation']
    train = passive + controls
    if not passive or not val:
        raise ValueError('Both passive TRAIN and VAL required')
    if len({(r['q'].shape, r['rest'].shape) for r in train}) != 1:
        raise ValueError('TRAIN topology/time-grid mismatch')
    checkpoint = torch.load(a.resume, map_location=device, weights_only=False)
    if checkpoint['model_class'] != 'NeuralGreenOperator':
        raise ValueError('Controlled teacher requires load-linear Green architecture')
    model = make_model(checkpoint['model_class'], checkpoint['model_kwargs'], train[0]['rest']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    # Fresh optimizer: this is a declared new objective, not a resumed old run.
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    a.output.mkdir(parents=True, exist_ok=False)
    names = ['train_gaussian_control_forward.py', 'train_gaussian_forward.py', 'gaussian_operator.py', 'gaussian_probe_roles.py']
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in names}
    (a.output / 'provenance').mkdir()
    for name, content in sources.items():
        (a.output / 'provenance' / name).write_bytes(content)
    (a.output / 'roles.json').write_bytes(a.roles.read_bytes())
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    catalog = [{k:r[k] for k in ['path', 'episode_id', 'split', 'group_id', 'trajectory_sha256', 'recipe_sha256', 'role']} for r in records + controls]
    (a.output / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    protocol = {k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    protocol.update(source_sha256=hashes, role_sha256=sha(a.roles), resume_sha256=sha(a.resume),
        passive_train=len(passive), controlled_train=len(controls), validation=len(val), no_test_labels_opened=True,
        selection='Primary final.pt at predeclared fixed step; best-passive-val.pt is a secondary diagnostic only',
        supervision='TRAIN physical labels and saved nodal trajectories; 0.4mm floor normalized MSE; Bernoulli passive/control sampling',
        excluded='Calibration-only controls; independent diagonal controls; reset replays; all VAL/TEST controls',
        runtime='Direct time neural evaluation; no FEM/integration')
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rest = torch.tensor(np.stack([r['rest'] for r in train]), device=device)
    q = torch.tensor(np.stack([r['q'] for r in train]), device=device)
    physical = torch.tensor(np.stack([r['physical'] for r in train]), device=device)
    bc = torch.tensor([r['bc'] for r in train], device=device)
    force = torch.tensor([r['force'] for r in train], device=device)
    scales = torch.tensor([r['scale_m'] for r in train], device=device)
    times = torch.tensor(train[0]['times'], device=device)
    started, best, history = time.monotonic(), float('inf'), []
    for step in range(1, a.steps + 1):
        model.train()
        is_control = rng.random(a.batch) < a.probe_fraction
        ids = np.where(is_control, len(passive) + rng.integers(len(controls), size=a.batch), rng.integers(len(passive), size=a.batch))
        ids = torch.tensor(ids, device=device)
        nodes = torch.tensor(rng.choice(rest.shape[1], min(a.points, rest.shape[1]), replace=False), device=device)
        tids = torch.tensor(np.sort(rng.choice(np.arange(1,len(times)), min(a.times,len(times)-1), replace=False)), device=device)
        predicted = model(rest[ids][:,nodes], times[tids], physical[ids], bc[ids], force[ids])
        loss = ((predicted-q[ids][:,tids][:,:,nodes])/scales[ids,None,None,None]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite controlled training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(step=step, loss=float(loss.detach()), elapsed_s=time.monotonic()-started)), flush=True)
        if step == 1 or step % a.eval_every == 0 or step == a.steps:
            validation = evaluate(model, val, device)
            fit_controls = evaluate(model, controls, device)
            history.append(dict(step=step, passive_validation_rmse_m=validation['mean_coordinate_rmse_m'],
                fit_control_rmse_m=fit_controls['mean_coordinate_rmse_m'], seconds=time.monotonic()-started))
            saved = dict(model_class=checkpoint['model_class'], model_kwargs=checkpoint['model_kwargs'], model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(), step=step, validation=validation, train_control_diagnostic=fit_controls,
                source_sha256=hashes, catalog=catalog, protocol=protocol)
            torch.save(saved, a.output / 'latest.pt')
            if validation['mean_coordinate_rmse_m'] < best:
                best = validation['mean_coordinate_rmse_m']
                torch.save(saved, a.output / 'best-passive-val.pt')
            if step == a.steps:
                torch.save(saved, a.output / 'final.pt')
            (a.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(history[-1]), flush=True)
            if any(Path(__file__).with_name(name).read_bytes() != content for name,content in sources.items()):
                raise RuntimeError('Source changed during training; preserve run and restart')


if __name__ == '__main__':
    main()
