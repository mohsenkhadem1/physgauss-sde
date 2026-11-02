"""Outer inference-context routing for the never-started cohort image study.
No original source mutation; numerical settings and all prediction gates retained.
A separate reporting amendment must pin this wrapper before preparation/execution.
"""
import argparse,importlib.util,json,os
from pathlib import Path
from scripts.compositional_dynamics_v17 import cohort_appearance as original
ROOT=original.ROOT
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort-appearance-inference'
AMENDMENT=ROOT/'reports/compositional-dynamics-v17/cohort-appearance-inference-amendment.json'

def bound():
    amendment=original.read(AMENDMENT)
    for name,h in amendment['source_sha256'].items():
        if original.sha(ROOT/name)!=h:raise ValueError('Inference routing source changed')
    if amendment['source_sha256'].get(str(Path(__file__).relative_to(ROOT)))!=original.sha(__file__):raise ValueError('Wrapper must be pinned')
    if original.BASE.exists():raise ValueError('Original cohort image run unexpectedly exists; no silent recovery')
    spec=importlib.util.spec_from_file_location('_cohort_inference',Path(original.__file__));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.BASE=BASE
    return m

def prepare(bindings):bound().prepare(bindings)

def predict():
    m=bound();os.environ['CUDA_VISIBLE_DEVICES']='1'
    import torch
    with torch.no_grad():m.predict()
    original.write(BASE/'inference-context-complete.json',dict(amendment_sha256=original.sha(AMENDMENT),prediction_commit_sha256=original.sha(BASE/'prediction-commit.json'),scope='Outer no_grad includes opacity filtering; same frozen rendering inputs/transport.'))

def score():
    m=bound();seal=original.read(BASE/'inference-context-complete.json')
    if seal['amendment_sha256']!=original.sha(AMENDMENT)or seal['prediction_commit_sha256']!=original.sha(BASE/'prediction-commit.json'):raise ValueError('Inference output seal changed')
    source=ROOT/'scripts/compositional_dynamics_v17/cohort_appearance_score.py';spec=importlib.util.spec_from_file_location('_cohort_inference_score',source);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);s.r=m;s.score()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','predict','score']);a.add_argument('--bindings',type=Path);args=a.parse_args();prepare(json.loads(args.bindings.read_text()))if args.stage=='prepare'else globals()[args.stage]()
