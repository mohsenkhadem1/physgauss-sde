"""Explicit hash-bound admission for twelve prospective SoftVT validation cases.

No production protocol is created here. Unknown, calibration and confirmation
roles are rejected before HDF5 access. Predictor reads a strict field whitelist.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import SOFTVT,_softvt_path,validate_predictor,predictor_digest
from scripts.prefix_ridge_switch_v13.inputs import softvt_payload

CASES=tuple((suite,task,episode)for suite in ('object-soft','spatial-soft')for task in range(3)for episode in (50,51))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def admission(subset,task,episode,*,protocol_path,protocol_sha256,root=SOFTVT):
    if (subset,task,episode)not in CASES:raise PermissionError('Only the twelve explicitly proposed validation cases may be admitted')
    path=Path(protocol_path)
    if not re.fullmatch('[0-9a-f]{64}',protocol_sha256)or sha(path)!=protocol_sha256:raise PermissionError('Validation protocol hash mismatch')
    protocol=json.loads(path.read_text())
    if protocol.get('schema')!=1 or protocol.get('status')!='frozen' or protocol.get('role')!='validation':raise PermissionError('Validation protocol is not frozen')
    rows=protocol.get('cases',[])
    if len(rows)!=12 or {(r['subset'],r['task'],r['episode'])for r in rows}!=set(CASES):raise PermissionError('Protocol must name exactly the twelve validation cases')
    for row in rows:
        if row.get('role')!='validation'or row.get('case')!=f"{row['subset']}-task{row['task']}-demo{row['episode']}":raise PermissionError('Invalid case role or identity')
    if not re.fullmatch('[0-9a-f]{64}',protocol.get('recipe_sha256','')):raise PermissionError('Frozen recipe digest required')
    recipe_path=Path(protocol.get('recipe_path',''))
    if not recipe_path.is_file()or sha(recipe_path)!=protocol['recipe_sha256']:raise PermissionError('Frozen recipe file digest mismatch')
    sources=protocol.get('source_sha256',{})
    if not sources or Path(__file__).resolve()not in {Path(name).resolve()for name in sources}:raise PermissionError('Frozen admission-adapter source digest required')
    recipe_entries=[digest for name,digest in sources.items()if Path(name).resolve()==recipe_path.resolve()]
    if recipe_entries!=[protocol['recipe_sha256']]:raise PermissionError('Recipe file must be bound once in source manifest')
    for name,digest in sources.items():
        if sha(name)!=digest:raise PermissionError('Changed frozen source: '+name)
    source=_softvt_path(subset,task,root)
    declared=protocol['source_files'][f'{subset}/task{task}'];stat=source.stat()
    if Path(declared['path']).resolve()!=source.resolve() or declared['size_bytes']!=stat.st_size or declared['mtime_ns']!=stat.st_mtime_ns:raise PermissionError('Source file metadata differs from frozen protocol')
    ledger=Path(protocol['access_ledger'])
    if not ledger.is_absolute():raise PermissionError('Absolute access-ledger path required')
    return protocol,source


def record(protocol,protocol_sha256,case,phase,fields,**details):
    path=Path(protocol['access_ledger']);path.parent.mkdir(parents=True,exist_ok=True)
    entry=dict(unix=time.time(),pid=os.getpid(),case=case,role='validation',phase=phase,fields=fields,protocol_sha256=protocol_sha256,**details)
    with path.open('a')as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(entry,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)


def load_predictor(subset,task,episode,*,protocol_path,protocol_sha256,root=SOFTVT):
    protocol,path=admission(subset,task,episode,protocol_path=protocol_path,protocol_sha256=protocol_sha256,root=root)
    case=f'{subset}-task{task}-demo{episode}'
    record(protocol,protocol_sha256,case,'predictor_read_start',['object nodal_pos_w[:floor(.6*T)]','eef_pose[:]','initial robot root_pose[0]','gripper_width[:]','gripper_binary[:]'],source=str(path))
    import h5py
    with h5py.File(path)as f:
        g=f[f'data/demo_{episode}'];asset=g.attrs['asset_name'];T=len(g['obs/eef_pose']);end=int(.6*T)
        raw=dict(object_prefix=g[f'states/deformable_object/{asset}/nodal_pos_w'][:end],eef_pose=g['obs/eef_pose'][:],
                 robot_root=g['initial_state/articulation/robot/root_pose'][0],width=g['obs/gripper_width'][:],command=g['obs/gripper_binary'][:])
    data=softvt_payload(raw,subset,task,episode,str(path),end)
    data.update(data_role='prospective_validation',validation_protocol_path=str(Path(protocol_path).resolve()),validation_protocol_sha256=protocol_sha256,
                validation_recipe_sha256=protocol['recipe_sha256'],observation_contract='Privileged initial volume and sparse prefix nodes; no RGB or force channels',
                boundary_contract='Initialized proximity priors, not true contact labels',units='SI metres and seconds')
    validate_predictor(data)
    record(protocol,protocol_sha256,case,'predictor_exported',[],prefix_end=end,total_frames=T,predictor_digest=predictor_digest(data))
    return data


def write_validation_package(data,output,protocol_sha256):
    from scripts.gaussian_boundary_flow_v02.export_inputs import write_package
    if data.get('data_role')!='prospective_validation'or data.get('validation_protocol_sha256')!=protocol_sha256:raise ValueError('Validation package metadata mismatch')
    manifest=write_package(data,Path(output))
    manifest.update(scope='Prospective validation package; prefix observations only, not historically exposed development',data_role='prospective_validation',validation_protocol_sha256=protocol_sha256)
    (Path(output)/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--subset',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--nodes',type=int,default=64);args=p.parse_args()
    from scripts.gaussian_boundary_flow_v02.export_inputs import reduce_geometry,write_package
    from scripts.gaussian_boundary_flow_v02.affine_observation import affine_observation_matrix
    data=load_predictor(args.subset,args.task,args.episode,protocol_path=args.protocol,protocol_sha256=args.protocol_sha256)
    if len(data['nodes'])>args.nodes:data=reduce_geometry(data,args.nodes)
    data['observation_matrix'],diagnostic=affine_observation_matrix(data['nodes'],data['observed_initial'])
    data['observation_map_revision']={'rule':'Initial-only affine-complete local map','maximum_row_l1':diagnostic['maximum_row_l1'],'maximum_row_l2':diagnostic['maximum_row_l2'],'negative_weights_allowed':True}
    write_validation_package(data,Path(args.output),args.protocol_sha256)
    protocol,_=admission(args.subset,args.task,args.episode,protocol_path=args.protocol,protocol_sha256=args.protocol_sha256)
    record(protocol,args.protocol_sha256,data['name'],'stable_package_written',[],package=str(Path(args.output).resolve()),manifest_sha256=sha(Path(args.output)/'manifest.json'),predictor_digest=predictor_digest(data))

if __name__=='__main__':main()
