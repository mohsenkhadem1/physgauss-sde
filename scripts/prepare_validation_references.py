"""Optional deterministic CPU partition for independent validation references."""
import argparse
import json
from pathlib import Path
import torch
from physgauss.calibrated_study import case_condition
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import production_refinement
from physgauss.stochastic_reference import coupled_reference


def prepare(study, partition=0, partitions=1):
    study = Path(study); manifest = json.loads((study/"manifest.json").read_text())
    plan = manifest["plan"]; torch.set_num_threads(plan["threads"])
    times = torch.linspace(0., 1., 21, dtype=torch.float64)
    seeds = plan["interval_calibration_seeds"]+plan["interval_test_seeds"]
    for seed in seeds[partition::partitions]:
        if source_fingerprint() != manifest["source_fingerprint_sha256"]: raise RuntimeError("Source changed.")
        condition = case_condition(seed)
        folder = study/"datasets"/f"{condition}__{seed}"; folder.mkdir(parents=True, exist_ok=True)
        path = folder/"reference.pt"
        if path.exists(): continue
        print("Preparing independent validation reference", condition, seed, flush=True)
        reference = coupled_reference(times, seed, condition == "pulse_and_stochastic", plan["reference_substeps"], plan["reference_last_change_tolerance_m"])
        reference["audit"]["production_staggered_refinement"] = production_refinement(plan, reference, condition == "pulse_and_stochastic")
        temporary = folder/f"reference.partition_{partition}.pt.tmp"; torch.save(reference, temporary)
        if path.exists():
            existing = torch.load(path, weights_only=True)
            torch.testing.assert_close(existing["positions"], reference["positions"], atol=0, rtol=0)
            if existing["audit"] != reference["audit"]: raise ValueError("Reference partition mismatch.")
            temporary.unlink()
        else: temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--partition", type=int, default=0); parser.add_argument("--partitions", type=int, default=1)
    args = parser.parse_args(); prepare(args.study, args.partition, args.partitions)
