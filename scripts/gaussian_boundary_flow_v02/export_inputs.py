"""Stable, pickle-free predictor packages across numerical environments.

Export once with the dataset-capable environment, then load exact persisted
arrays rather than rebuilding Delaunay interpolation in another SciPy version.
No evaluator arrays or fresh cases are opened by this module's export commands.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile
import numpy as np

FORMAT_VERSION=1
ARRAY_FILE='predictor.npz'
META_FILE='predictor-meta.json'
MANIFEST_FILE='manifest.json'
PHYSTWIN_SMOKE=('single_lift_sloth','single_push_sloth','single_lift_zebra','double_lift_zebra')
SOFTVT_SMOKE=tuple((suite,task,0)for suite in ('object-soft','spatial-soft')for task in (1,2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):_json_value(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_value(v)for v in value]
    return value


def reduce_geometry(data,count=None):
    """FPS reference nodes; preserve full geometry, observations and command clock.

    Only nodal controller targets and attachment weights are subset. The point
    observation operator is rebuilt ONCE here using the original point locations.
    All non-geometric inputs remain unchanged. Never mutate the caller's payload.
    """
    from scripts.gaussian_boundary_flow_v02.benchmark_inputs import validate_predictor,predictor_digest
    from scripts.external_benchmark.data import fps,interpolate_matrix
    validate_predictor(data)
    result={k:v.copy()if isinstance(v,np.ndarray)else _json_value(v)for k,v in data.items()}
    if count is None:return result
    if not isinstance(count,int)or count<4:raise ValueError('At least four reference nodes required')
    original=np.asarray(data['nodes']);n=len(original)
    result['support_geometry']=np.asarray(data.get('support_geometry',original)).copy()
    ids=fps(original,min(count,n))if count<n else np.arange(n)
    result['nodes']=original[ids].copy()
    targets=np.asarray(data['controller_targets']);weights=np.asarray(data['attachment_weights'])
    if targets.ndim!=4 or targets.shape[2]!=n or weights.ndim!=2 or weights.shape[1]!=n:
        raise ValueError('Unexpected nodal controller/attachment layout')
    result['controller_targets']=targets[:,:,ids,:].copy()
    result['attachment_weights']=weights[:,ids].copy()
    if count<n:
        result['observation_matrix']=interpolate_matrix(result['nodes'],data['observed_initial'])
    result['input_geometry_reduction']=dict(method='Initial-geometry FPS',original_nodes=n,
        retained_nodes=len(ids),source_node_ids=ids.tolist(),original_predictor_digest=predictor_digest(data),
        support_geometry='Original full initial support geometry retained; no future geometry read')
    return validate_predictor(result)


def write_package(data,destination,*,reduction=None):
    from scripts.gaussian_boundary_flow_v02.benchmark_inputs import validate_predictor,predictor_digest
    validate_predictor(data)
    arrays={k:np.asarray(v)for k,v in data.items()if isinstance(v,np.ndarray)}
    if any(a.dtype.hasobject for a in arrays.values()):raise ValueError('Object arrays require pickle and are forbidden')
    meta={k:_json_value(v)for k,v in data.items()if k not in arrays}
    # Canonical metadata round-trip is part of the stored payload's contract.
    meta=json.loads(json.dumps(meta,sort_keys=True,allow_nan=False))
    normalized=dict(meta,**arrays);digest=predictor_digest(normalized)
    destination=Path(destination)
    if destination.exists():
        previous=load_package(destination)
        if predictor_digest(previous)!=digest:raise ValueError('Existing immutable package has different predictor content')
        return json.loads((destination/MANIFEST_FILE).read_text())
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix='.export-')as temporary:
        tmp=Path(temporary)
        np.savez_compressed(tmp/ARRAY_FILE,**arrays)
        (tmp/META_FILE).write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
        manifest=dict(format_version=FORMAT_VERSION,name=normalized['name'],dataset=normalized['dataset'],
            arrays_file=ARRAY_FILE,metadata_file=META_FILE,
            arrays_sha256=sha(tmp/ARRAY_FILE),metadata_sha256=sha(tmp/META_FILE),predictor_digest=digest,
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in arrays.items()},
            observed_frames=int(normalized['train_end']),action_frames=len(normalized['times']),nodes=len(normalized['nodes']),
            reduction=reduction,source=normalized.get('source'),evaluation_exported=False,force_labels_exported=False,
            scope='Historically exposed development cases only; no evaluator targets or fresh trajectories',
            exporter_environment=dict(python=platform.python_version(),numpy=np.__version__))
        try:
            import scipy
            manifest['exporter_environment']['scipy']=scipy.__version__
        except ImportError:pass
        (tmp/MANIFEST_FILE).write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
        # Validate exactly the bytes about to be promoted.
        load_package(tmp)
        os.replace(tmp,destination)
    return manifest


def load_package(directory):
    """No raw dataset, HDF5, pickle, FPS or triangulation accesses at reload time."""
    from scripts.gaussian_boundary_flow_v02.benchmark_inputs import validate_predictor,predictor_digest
    directory=Path(directory);manifest=json.loads((directory/MANIFEST_FILE).read_text())
    if manifest.get('format_version')!=FORMAT_VERSION:raise ValueError('Unknown predictor package version')
    if manifest.get('arrays_file')!=ARRAY_FILE or manifest.get('metadata_file')!=META_FILE:
        raise ValueError('Invalid predictor package filenames')
    if manifest.get('evaluation_exported')is not False or manifest.get('force_labels_exported')is not False:
        raise ValueError('Package contains undeclared privileged outputs')
    if sha(directory/ARRAY_FILE)!=manifest['arrays_sha256']or sha(directory/META_FILE)!=manifest['metadata_sha256']:
        raise ValueError('Predictor package file hash mismatch')
    metadata=json.loads((directory/META_FILE).read_text())
    with np.load(directory/ARRAY_FILE,allow_pickle=False)as archive:
        if set(archive.files)!=set(manifest['arrays']):raise ValueError('Array inventory mismatch')
        arrays={k:archive[k].copy()for k in archive.files}
    if set(metadata)&set(arrays):raise ValueError('Metadata/array key collision')
    for key,value in arrays.items():
        expected=manifest['arrays'][key]
        if value.dtype.hasobject or list(value.shape)!=expected['shape']or str(value.dtype)!=expected['dtype']:
            raise ValueError('Array shape/dtype mismatch: '+key)
    result=dict(metadata,**arrays);validate_predictor(result)
    if predictor_digest(result)!=manifest['predictor_digest']:raise ValueError('Predictor semantic digest mismatch')
    return result


def export_smoke(output_root,reduce_nodes=64):
    # Dataset-specific imports stay inside export; reload never needs h5py.
    from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor,load_softvt_predictor
    output_root=Path(output_root);rows=[]
    for case in PHYSTWIN_SMOKE:
        data=load_phystwin_predictor(case,max_nodes=120)
        reduced=reduce_geometry(data,reduce_nodes);path=output_root/case
        manifest=write_package(reduced,path,reduction=reduce_nodes)
        rows.append(dict(path=str(path),**manifest))
    for suite,task,episode in SOFTVT_SMOKE:
        data=load_softvt_predictor(suite,task,episode)
        reduced=reduce_geometry(data,reduce_nodes);path=output_root/reduced['name']
        manifest=write_package(reduced,path,reduction=reduce_nodes)
        rows.append(dict(path=str(path),**manifest))
    return dict(packages=rows,fresh_trajectories_read=False,model_executed=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',default='runs/gaussian-boundary-flow-v02-inputs');parser.add_argument('--reduce-nodes',type=int,default=64)
    parser.add_argument('--report',default='reports/gaussian-boundary-flow-v02/stable-input-export.json');args=parser.parse_args()
    report=export_smoke(args.output_root,args.reduce_nodes);path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({r['name']:{'nodes':r['nodes'],'observed_frames':r['observed_frames'],'predictor_digest':r['predictor_digest']}for r in report['packages']},indent=2))

if __name__=='__main__':main()
