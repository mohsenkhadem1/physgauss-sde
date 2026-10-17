"""Prepare independent stochastic references beside an active matching study.

Only reference files are written, atomically. Fitting and image generation
remain in the main runner. This optional CPU helper uses exactly the same
committed functions/seeds as serial execution, not another experiment.
"""
import argparse
import json
from pathlib import Path
import torch
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import CONDITIONS, production_refinement
from physgauss.stochastic_reference import coupled_reference


def prepare(root):
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    plan = manifest["plan"]; torch.set_num_threads(plan["threads"])
    times = torch.linspace(0., 1., 21, dtype=torch.float64)
    for condition in plan["conditions"]:
        pulse, stochastic = CONDITIONS[condition]
        if not stochastic: continue
        for seed in plan["seeds"]:
            if source_fingerprint() != manifest["source_fingerprint_sha256"]:
                raise RuntimeError("Numerical source changed; do not mix references.")
            folder = root/"datasets"/f"{condition}__{seed}"; folder.mkdir(parents=True, exist_ok=True)
            target = folder/"reference.pt"
            if target.exists(): continue
            print(f"Preparing coupled reference: {condition}__{seed}", flush=True)
            reference = coupled_reference(times, seed, pulse, plan["reference_substeps"], plan["reference_last_change_tolerance_m"])
            reference["audit"]["production_staggered_refinement"] = production_refinement(plan, reference, pulse)
            temporary = folder/"reference.helper.pt.tmp"
            torch.save(reference, temporary)
            if target.exists():
                existing = torch.load(target, weights_only=True)
                torch.testing.assert_close(existing["positions"], reference["positions"], rtol=0, atol=0)
                if existing["audit"] != reference["audit"]: raise ValueError("Concurrent reference mismatch.")
            else:
                temporary.replace(target)
            print(f"Prepared: {condition}__{seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(); prepare(args.study)
