"""Resumable FEM bank and sealed object splits for continuous uncertainty diagnosis."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.controlled_reference import ControlledBlock, ACTIONS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/phystwin-uncertainty-v02'
BOUNDS = np.log([[1500., 10000.], [40., 1600.]])


def parameters(count, seed):
    rng = np.random.default_rng(seed)
    return np.exp(BOUNDS[:, 0] + rng.random((count, 2)) * np.diff(BOUNDS, axis=1).ravel())


def design():
    return dict(version=2, seed=20261028, bounds_E_Pa=[1500, 10000], bounds_K_N_m=[40, 1600],
        prior='uniform in log E and log K', bank_mesh_n=4, bank_grid=9,
        truth_mesh_n=8, development=parameters(12, 101).tolist(),
        calibration=parameters(20, 102).tolist(), test=parameters(24, 103).tolist(),
        sigma_m=.0003, noise_repeats=4, quadrature_sizes=[65, 129, 257],
        quadrature_gate=dict(mean_m=.000005, sd_m=.000005, log_parameter_quantile=.01),
        interpolation_gate_rms_m=.00001,
        discrepancy='development mean residual plus sample covariance, 0.1 shrinkage to its diagonal and 1 micrometre diagonal floor',
        intervals='Gaussian moment intervals, explicitly not exact mixture quantiles; separately object-level split-conformal scaling',
        conformal='90% simultaneous field coverage across all coordinates, four noise repeats and all three observation budgets; 19th of 20 calibration scores',
        policies=['initial end only', 'end plus midpoint axial', 'end plus midpoint sideways'],
        target='complete 81-point oblique displacement field',
        adaptive_gate='No adaptive superiority claim; first audit uncertainty using fixed actions and matched random average',
        scope='Same block geometry and fixture form; new material/support pairs, not new layouts or real images')


def frozen_protocol():
    BASE.mkdir(parents=True, exist_ok=True)
    p = BASE / 'protocol.json'
    expected = design()
    if p.exists():
        assert json.loads(p.read_text()) == expected, 'Frozen protocol changed'
    else:
        p.write_text(json.dumps(expected, indent=2) + '\n')
    return expected


def solve_job(job):
    name, E, K, n = job
    path = BASE / 'fields' / (name + '.npz')
    if path.exists():
        with np.load(path) as z:
            assert np.allclose(z['parameters'], [E, K]) and int(z['n']) == n
        return name, 'cached'
    fields, residuals, jacobians = [], [], []
    for action in ACTIONS:
        model = ControlledBlock(n, E, K)
        u = np.zeros_like(model.rest)
        for alpha in np.linspace(0, 1, 9)[1:]:
            model.target = ACTIONS[action] * alpha
            answer = model.solve(u)
            u = answer['displacement_local_m']
            residuals.append(float(np.linalg.norm(answer['residual'])))
            jacobians.append(float(answer['J'].min()))
        fields.append(u.reshape(4*n+1, n+1, n+1, 3)[::n//2, ::n//2, ::n//2].reshape(81, 3))
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, fields=np.array(fields), parameters=[E, K], n=n,
                        max_residual_N=max(residuals), minimum_J=min(jacobians))
    tmp.replace(path)
    return name, 'solved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split', choices=['bank', 'development', 'calibration', 'test', 'all'], default='all')
    parser.add_argument('--bank-grid', type=int, choices=[9, 17], default=9)
    args = parser.parse_args()
    spec = frozen_protocol()
    (BASE / 'fields').mkdir(exist_ok=True)
    jobs = []
    if args.split in ['bank', 'all']:
        axes = [np.linspace(*b, args.bank_grid) for b in BOUNDS]
        for i, e in enumerate(axes[0]):
            for j, k in enumerate(axes[1]):
                jobs.append((f'bank{args.bank_grid}-{i}-{j}', np.exp(e), np.exp(k), 4))
    for split in ['development', 'calibration', 'test']:
        if args.split not in [split, 'all']:
            continue
        for i, (E, K) in enumerate(spec[split]):
            jobs.append((f'{split}-{i}-n8', E, K, 8))
            if split == 'development':
                jobs.append((f'{split}-{i}-n4', E, K, 4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(solve_job, jobs)):
            print(i+1, len(jobs), *result, flush=True)

if __name__ == '__main__':
    main()
