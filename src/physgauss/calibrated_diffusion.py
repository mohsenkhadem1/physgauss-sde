"""Operating-region diffusion calibration, with trajectory-disjoint supervision.

Only independent synthetic innovation observations enter training. Analytic
amplitudes are used by the simulator and evaluation diagnostics, not the loss.
No sparse-view evaluation trajectory is read by this module.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from .physics import cube_mesh, edge_features, simulate
from .stochastic import spatial_basis, calibration_law
from .study import write_json


class CalibratedDiffusion(nn.Module):
    """33 learned parameters; symmetric, scaled mechanical invariant inputs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)).double()
        self.register_buffer("feature_scale", torch.tensor([.10, .50], dtype=torch.float64))

    def forward(self, features):
        # Smooth magnitude enforces sign symmetry without a derivative kink.
        magnitude = (features.square()+.001**2).sqrt()/self.feature_scale
        return .001*torch.nn.functional.softplus(self.net(magnitude).squeeze(-1))+.0001


@torch.no_grad()
def trajectory_features(seeds, substeps=48):
    """Varied material/initial states, independent Brownian paths, known simulator.

    These trajectories supply calibration states, not reconstruction targets.
    A modest grid suffices to sample the operating region; it is not used as
    accurate evaluation ground truth. Both pulse and no-pulse examples occur.
    """
    class Oracle(nn.Module):
        def forward(self, value): return calibration_law(value)
    mesh = cube_mesh(); times = torch.linspace(0., 1., 21, dtype=torch.float64)
    basis = spatial_basis(mesh, 3, .45)
    groups, metadata = [], []
    for index, seed in enumerate(seeds):
        generator = torch.Generator().manual_seed(seed)
        draw = torch.rand(8, dtype=times.dtype, generator=generator)
        young = 12.+16*draw[0:1]; drag = .3+.6*draw[1:2]
        offset = (draw[2:5]-.5)*times.new_tensor([.26, .12, .18])
        velocity = (draw[5:8]-.5)*.5
        white = torch.randn(1, 20*substeps, 3, dtype=times.dtype, generator=generator)
        pulse = index % 2 == 1
        def external(t, x, v):
            amplitude = math.exp(-.5*((t-.5)/.08)**2) if pulse else 0.
            return amplitude*(~mesh.pinned)[None, :, None]*x.new_tensor([.45, 0., -.30])
        path = simulate(mesh, times, young, drag, offset[None], velocity[None],
                        substeps=substeps, white=white, diffusion=Oracle(), noise_basis=basis,
                        external_acceleration=external)
        features = edge_features(path.x, path.v, mesh)[0].reshape(-1, 2)
        groups.append(features)
        metadata.append({"seed": seed, "young_pa": float(young[0]), "drag_per_s": float(drag[0]),
            "offset_m": offset.tolist(), "initial_velocity_m_per_s": velocity.tolist(),
            "pulse": pulse, "feature_pairs": len(features),
            "features_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest()})
    return torch.cat(groups), metadata


def amplitude_metrics(network, features, innovations_seed=74001, replicates=64):
    with torch.no_grad():
        truth = calibration_law(features); prediction = network(features)
        generator = torch.Generator().manual_seed(innovations_seed)
        standardized = truth[:, None]*torch.randn(len(truth), replicates, dtype=truth.dtype,
                                                  generator=generator)/prediction[:, None]
        return {"relative_sigma_rmse": float((prediction/truth-1).square().mean().sqrt()),
            "mean_prediction": float(prediction.mean()), "mean_true_amplitude": float(truth.mean()),
            "prediction_cv": float(prediction.std()/prediction.mean()), "truth_cv": float(truth.std()/truth.mean()),
            "standardized_innovation_mean": float(standardized.mean()),
            "standardized_innovation_variance": float(standardized.var()),
            "innovation_90_coverage": float((standardized.abs() <= 1.6448536269514722).double().mean()),
            "features": len(features), "replicates_per_feature": replicates,
            "qualification": "correlated states within independent trajectories; not independent feature-level replicates"}


def train_calibrated(settings, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        record = json.loads(destination.read_text())
        if record["settings"] != settings: raise ValueError("Calibration settings changed.")
        network = CalibratedDiffusion()
        network.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in record["state_dict"].items()})
    else:
        if set(settings["training_seeds"]) & set(settings["validation_seeds"]):
            raise ValueError("Training and validation trajectories must be disjoint.")
        operating, train_rows = trajectory_features(settings["training_seeds"], settings["substeps"])
        validation, validation_rows = trajectory_features(settings["validation_seeds"], settings["substeps"])
        generator = torch.Generator().manual_seed(settings["innovation_seed"])
        # Equal mass on operating-region trajectories and broad exploratory states.
        broad = torch.rand(len(operating), 2, dtype=operating.dtype, generator=generator)
        broad = broad*broad.new_tensor([.7, 3.])-broad.new_tensor([.25, 1.5])
        features = torch.cat((operating, broad))
        innovations = calibration_law(features)[:, None]*torch.randn(len(features), settings["replicates"],
            dtype=features.dtype, generator=generator)
        empirical_second_moment = innovations.square().mean(1)
        with torch.random.fork_rng():
            torch.manual_seed(settings["network_seed"]); network = CalibratedDiffusion()
        optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
        history = []
        for step in range(settings["steps"]):
            sigma = network(features)
            loss = (.5*empirical_second_moment/sigma.square()+sigma.log()).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if (step+1) % 250 == 0 or step+1 == settings["steps"]:
                history.append({"step": step+1, "innovation_nll": float(loss.detach())})
        # Fixed-strength control fitted directly to the SAME training innovations,
        # rather than copying potentially miscalibrated network predictions.
        constant = float(empirical_second_moment[:len(operating)].mean().sqrt())
        class Constant(nn.Module):
            def forward(self, value): return value.new_full(value.shape[:-1], constant)
        record = {"settings": settings, "architecture": "smooth absolute mechanical invariants / [.10,.50] -> Linear(2,8) -> tanh -> Linear(8,1) -> .001 softplus + .0001",
            "learned_parameters": sum(p.numel() for p in network.parameters()),
            "training_trajectories": train_rows, "validation_trajectories": validation_rows,
            "training_feature_pairs": len(features), "training_history": history,
            "constant_amplitude": constant,
            "constant_definition": "Gaussian innovation MLE on training operating-region innovations only",
            "validation": amplitude_metrics(network, validation, settings["validation_innovation_seed"]),
            "constant_validation": amplitude_metrics(Constant(), validation, settings["validation_innovation_seed"]),
            "state_dict": {k: v.detach().tolist() for k, v in network.state_dict().items()},
            "qualification": "independent synthetic simulator supervision, frozen before sparse reconstruction; no test trajectory or withheld RGB used"}
        write_json(destination, record)
    network.eval()
    for parameter in network.parameters(): parameter.requires_grad_(False)
    return network, record
