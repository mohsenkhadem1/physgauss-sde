"""Create private held worlds only from a frozen protocol; never read fit results.

Private material values remain under runs/. Public action recipes and identifiers
are exported separately. Existing campaigns cannot be overwritten or resampled.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from dataclasses import asdict
from scripts.complex_scenes_v18.world_queries import Material, World
from scripts.complex_scenes_v18.run_development import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_freeze(freeze, root):
    if freeze.get('status')!='frozen' or freeze.get('held_data_generated'):
        raise ValueError('A prospective frozen protocol is required')
    if not freeze.get('source_sha256') or not freeze.get('artifact_sha256'):
        raise ValueError('Source and model inventory required')
    for group in ('source_sha256','artifact_sha256'):
        for name,digest in freeze[group].items():
            path=Path(root)/name
            if not path.is_file() or sha(path)!=digest:raise ValueError('Frozen input changed: '+name)
    return freeze['protocol']


def create(freeze_path, output, seed, root):
    freeze_path=Path(freeze_path);output=Path(output)
    freeze=json.loads(freeze_path.read_text());protocol=validate_freeze(freeze,root)
    if output.exists():raise ValueError('Preserve existing held campaign')
    rng=np.random.default_rng(seed);distribution=protocol['material_distribution'];private=[];public=[]
    lo,hi=np.log(distribution['young_Pa_log_uniform'])
    for family in protocol['families']:
        for index in range(family['worlds']):
            name=f"{family['family']}-world{index:02d}";materials=[]
            for body in range(2 if family['family']=='deformable_pair'else 1):
                contrast=0.;transition=.5
                if family['family']=='heterogeneous':
                    contrast=float(rng.uniform(*np.log(distribution['heterogeneous_contrast_log_uniform'])))
                    transition=float(rng.uniform(*distribution['heterogeneous_transition_uniform']))
                materials.append(Material(float(rng.uniform(lo,hi)),float(rng.uniform(*distribution['poisson_uniform'])),contrast,transition))
            world=World(name,tuple(materials));private.append(asdict(world))
            public.append(dict(world_id=name,world_material_commitment=world.digest,family=family['family'],calibration=family['calibration'],calibration_duration_s=family['calibration_duration_s'],queries=family['queries'],refinement=index==0))
    if len(private)!=protocol['independent_worlds'] or sum(len(w['queries'])for w in public)!=protocol['total_queries']:raise ValueError('Protocol census mismatch')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'private-worlds.json',dict(seed=int(seed),worlds=private))
    write(output/'public-manifest.json',dict(split='held',freeze_sha256=sha(freeze_path),worlds=public))
    write(output/'manifest-seal.json',dict(freeze_sha256=sha(freeze_path),private_worlds_sha256=sha(output/'private-worlds.json'),public_manifest_sha256=sha(output/'public-manifest.json'),independent_worlds=len(private),total_queries=protocol['total_queries']))
    return output/'manifest-seal.json'


if __name__=='__main__':
    from scripts.complex_scenes_v18.run_development import ROOT
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    print(create(a.freeze,a.output,a.seed,ROOT))
