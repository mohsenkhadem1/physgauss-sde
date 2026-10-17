"""Controlled MAP ablation: calibration changes under the same final operator.

Development condition only (pulse+stochastic, three paired benchmark seeds).
All variants retain exact force marginalization and all density normalizers.
An oracle diffusion amplitude is an evaluation-only diagnostic, not a fitted
model available from real observations. This script is not posterior inference.
"""
import argparse
import json
from pathlib import Path
import torch
from physgauss.benchmark import Observations
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.neural_marginal_inference import fit_neural_marginal
from physgauss.marginal_inference import TrajectoryFitConfig
from physgauss.stochastic import calibration_law
from physgauss.study import write_json
from physgauss.cli import source_fingerprint
from physgauss.accurate_study import digest_tensor


class Oracle(torch.nn.Module):
    def forward(self, features):
        return calibration_law(features)


def run(output, seeds=(4101, 4102, 4103)):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    base = Path('examples/calibrated-noise/comparison')
    plan = json.loads((base/'manifest.json').read_text())['plan']
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled = True; backend.strategy = 'greedy'
    new_net, new_cal = calibrate(plan, base/'calibration.json')
    old_root = Path('examples/neural-marginal')
    old_plan = json.loads((old_root/'manifest.json').read_text())['plan']
    old_net, old_cal = calibrate(old_plan, old_root/'calibration.json')
    initial = torch.tensor(json.loads(Path('examples/calibrated-noise-setup/initial_state.json').read_text())['coordinates'], dtype=torch.float64)
    rows = []
    for seed in seeds:
        data = json.loads((base/'datasets'/f'pulse_and_stochastic__{seed}'/'inputs.json').read_text())
        truth = initial.new_tensor(data['positions_m_evaluation_only'])
        for label, net, cal, mode in [('legacy_constant', old_net, old_cal, 'constant'),
                ('legacy_neural', old_net, old_cal, 'neural'), ('oracle_amplitude', Oracle(), new_cal, 'neural')]:
            folder = output/f'{seed}__{label}'; folder.mkdir(parents=True, exist_ok=True)
            model = construct(plan, net, cal, mode)
            obs = Observations(model.view_frames, initial.new_tensor(data['observed_rgb']), .015)
            if (folder/'fit.pt').exists():
                fit = torch.load(folder/'fit.pt', weights_only=True)
            else:
                print(seed, label, flush=True)
                fit = fit_neural_marginal(model, obs, initial, TrajectoryFitConfig(**plan['optimizer']))
                torch.save(fit, folder/'fit.pt')
            with torch.no_grad():
                gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
                path = model.trajectory(fit['mean'][None]).x[0]
                error = path[:, model.free]-truth[:, model.free]
            record = {'seed': seed, 'variant': label, 'source_fingerprint_sha256': source_fingerprint(),
                'observations_sha256': digest_tensor(obs.rgb), 'initial_state_sha256': digest_tensor(initial),
                'gap_rmse_m': float(error[gap].square().mean().sqrt()), 'fit': fit['record'],
                'qualification': 'same final mechanics/renderer/kernel, calibration only changed; oracle uses analytic law diagnostically; all MAP estimates'}
            write_json(folder/'summary.json', record); rows.append(record)
        for label, mode in [('calibrated_constant', 'constant'), ('calibrated_neural', 'neural')]:
            old = json.loads((base/'cases'/f'pulse_and_stochastic__{seed}__{mode}'/'summary.json').read_text())
            rows.append({'seed': seed, 'variant': label, 'gap_rmse_m': old['scores']['original_gap']['position_rmse_m'], 'fit': old['fit'],
                'qualification': 'previous final MAP, same normalized target reproduced by the new sampler pipeline'})
    write_json(output/'summary.json', {'cases': rows, 'scope': 'three previously inspected pulse+stochastic benchmark seeds; diagnostic ablation, not an independent test',
        'means_m': {label: sum(r['gap_rmse_m'] for r in rows if r['variant'] == label)/len(seeds) for label in sorted({r['variant'] for r in rows})}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[4101, 4102, 4103])
    args = p.parse_args()
    run(args.output, [args.seed] if args.seed is not None else [4101, 4102, 4103])
