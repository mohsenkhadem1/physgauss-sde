"""Separately prespecified guarded-geometry videos; never overwrite primary."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.contact_benchmark_v12.render import render

CASES=[('regression',dict(case='double_stretch_sloth')),
       ('untouched_confirmation',dict(subset='object-soft',task=1,episode=10)),
       ('fresh_confirmation',dict(case='single_push_sloth')),
       ('fresh_confirmation',dict(subset='spatial-soft',task=2,episode=20))]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--case');args=p.parse_args()
    if not args.all and not args.case:p.error('Choose --all or --case NAME from prespecified cases')
    chosen=0
    for cohort,spec in CASES:
        name=spec.get('case',f"{spec.get('subset')}-task{spec.get('task')}-demo{spec.get('episode')}")
        if args.all or args.case==name:render(cohort,spec,guarded=True);chosen+=1
    if not chosen:p.error('Unknown prespecified case')
