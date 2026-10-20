"""Final whole-dataset replay, grouped split checks and deliberate corruption controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from isaac_dataset import make_plan
from validate_isaac_dataset import validate,sha,require


def audit(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());require(plan==make_plan(),'Frozen plan mismatch')
    cases=plan['episodes']
    def check(case):
        directory=root/'episodes'/case['episode_id'];result=validate(directory,case)
        (directory/'validation.json').write_text(json.dumps(result,indent=2)+'\n');return result
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,cases))
    groups={}
    for case in cases:groups.setdefault(case['group_id'],[]).append(case)
    require(all(len(group)==3 and len({c['split'] for c in group})==1 for group in groups.values()),'Material pairing/split leakage')
    controls=[]
    for corruption in ['camera_pose','force_command','boundary_target','source','extra_sparse_frame','recipe_split']:
        with tempfile.TemporaryDirectory(prefix='isaac-negative-',dir=root.parent) as temp:
            clone=Path(temp)/'episode';shutil.copytree(root/'episodes/e0000',clone)
            if corruption in ['force_command','boundary_target','camera_pose']:
                path=clone/('truth/cameras.npz' if corruption=='camera_pose' else 'truth/trajectory.npz')
                with np.load(path) as raw:arrays={k:raw[k] for k in raw.files}
                if corruption=='camera_pose':arrays['camera_to_world_usd_row'][0,3,0]+=.01
                elif corruption=='force_command':arrays['force_step_commands_N'][100,0,0]+=.01
                else:arrays['boundary_target_positions_m'][10,0,0]+=.001
                np.savez_compressed(path,**arrays)
            elif corruption=='source':
                path=clone/'provenance/isaac_dataset.py';path.write_bytes(path.read_bytes()+b'\n# altered\n')
            elif corruption=='extra_sparse_frame':shutil.copy2(clone/'truth/rgb/00021.png',clone/'observations/rgb/00021.png')
            else:
                path=clone/'truth/recipe.json';recipe=json.loads(path.read_text());recipe['split']='test';path.write_text(json.dumps(recipe))
            try:validate(clone,cases[0])
            except (ValueError,AssertionError) as error:controls.append({'corruption':corruption,'rejected':True,'reason':str(error)[:240]})
            else:raise RuntimeError(f'Integrity audit failed to reject {corruption}')
    names=['isaac_dataset.py','isaac_dataset_worker.py','run_isaac_dataset.py','validate_isaac_dataset.py',
           'describe_isaac_dataset.py','preview_isaac_dataset.py','audit_isaac_dataset.py','archive_isaac_dataset.py','boundary_block.py']
    record={'passed':True,'episodes_verified':len(results),'depth_rays_tested':sum(r['depth_rays_tested'] for r in results),
            'maximum_ray_depth_error_m':max(r['maximum_ray_depth_error_m'] for r in results),
            'maximum_load_patch_distance_error_m':max(r['load_patch_distance_error_m'] for r in results),
            'groups_disjoint_across_splits':True,'deliberate_corruption_controls':controls,
            'audit_source_sha256':{name:sha(Path(__file__).with_name(name)) for name in names}}
    (root/'audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['file-manifest.json','.generation.lock']]
    (root/'file-manifest.json').write_text(json.dumps({'files':files,'total_bytes':sum(f['bytes'] for f in files)},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset');args=parser.parse_args();audit(args.dataset)
