"""Freeze complete TRAIN/VAL supervision; TEST private data is never opened."""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse,json
from pathlib import Path
from train_gaussian_inverse import collect_cases,snapshot_catalog


def freeze(roots,output,expected_train=152,expected_validation=44):
    planned={};groups={}
    for root in map(Path,roots):
        plan=json.loads((root/'plan.json').read_text())
        for recipe in plan['episodes']:
            # Split/group metadata is public protocol information, including
            # TEST membership; no TEST observations, audit or truth is opened.
            if 'group_id' in recipe:
                group=(str(root.resolve()),recipe['group_id'])
                if group in groups and groups[group]!=recipe['split']:
                    raise ValueError('Layout group overlaps declared splits')
                groups[group]=recipe['split']
            if recipe['split'] not in ('train','validation'):continue
            key=(str(root.resolve()),recipe['episode_id'])
            if key in planned:raise ValueError('Duplicate root or planned episode')
            planned[key]=recipe['split']
    expected={'train':expected_train,'validation':expected_validation}
    counts={split:sum(s==split for s in planned.values()) for split in expected}
    if counts!=expected:raise ValueError(f'Planned split counts {counts} differ from required {expected}')
    completed=collect_cases(roots)
    available={(str(r['path'].parent.parent.resolve()),r['recipe']['episode_id']) for r in completed}
    if available!=set(planned):
        missing=sorted(set(planned)-available)
        raise ValueError(f'Cannot freeze incomplete supervision: {len(missing)} planned TRAIN/VAL episodes missing or failed: {missing[:8]}')
    return snapshot_catalog(roots,output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-train',type=int,default=152);p.add_argument('--expected-validation',type=int,default=44)
    a=p.parse_args();result=freeze(a.roots,a.output,a.expected_train,a.expected_validation)
    print(json.dumps({'catalog_sha256':result['catalog_sha256'],'cases':len(result['cases']),'no_test_cases':True},indent=2))
