"""Prospective Stage-A oracle composition/command protocol.

Reserved latent scenes are constructed only after a separate method freeze.
This stage evaluates forward models with identical oracle descriptors. Inverse
methods require their own untouched cohort; this is not an unknown-physics test.
"""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from scripts.compositional_neural_mechanics.scenes import split_cells,sample_scene,NOVEL_GEOMETRIES
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec


def required_sources(root):
    root=Path(root).resolve()
    pending=[Path(__file__).resolve(),Path(__file__).with_name('forward_evaluation.py').resolve(),Path(__file__).with_name('direct_baseline.py').resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module,*[node.module+'.'+x.name for x in node.names if x.name!='*']] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                if name.startswith(('scripts.','experiments.')):
                    candidate=root/(name.replace('.','/')+'.py')
                    if candidate.exists() and candidate not in seen:pending.append(candidate)
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(seen)}


def geometry_reference_cells():
    return [[cell for cell in split_cells()['train'] if cell[0]==g][index] for g in range(2) for index in (0,3,5)]


def held_commands(geometry):
    L,H,D=geometry.dimensions;commands={}
    p=np.zeros((2,6));p[1,5]=-.4;commands['negative_bend']=p
    p=np.zeros((2,6));p[1,4]=.4;commands['out_of_plane_bend']=p
    p=np.zeros((2,6));p[0,0]=-.07*L;p[1,0]=.07*L;p[0,3]=-.3;p[1,3]=.3;commands['stretch_twist']=p
    p=np.zeros((2,6));p[0,1]=-.08*H;p[1,1]=.08*H;p[0,2]=.08*D;p[1,2]=-.08*D;commands['biaxial_shear']=p
    return commands


def construct_reserved_records(freeze_path,root):
    root=Path(root);freeze=json.loads(Path(freeze_path).read_text())
    if freeze.get('status')!='frozen' or freeze.get('stage')!='oracle-forward-composition-v16':raise ValueError('Explicit oracle method freeze required before reserved construction')
    if freeze.get('seeds')!={'composition':926521,'geometry':926523}:raise ValueError('Prospectively declared reserved seeds required')
    if not freeze.get('source_sha256') or set(freeze.get('checkpoints',{}))!={'bulk','interface','direct','direct_long'}:raise ValueError('Frozen laws and both direct-baseline checkpoints required')
    required=required_sources(root)
    if any(freeze['source_sha256'].get(name)!=digest for name,digest in required.items()):raise ValueError('Complete recursive method source commitment required')
    for name,digest in freeze['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
    for item in freeze['checkpoints'].values():
        if hashlib.sha256((root/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Frozen checkpoint changed')
    rows=[];rng=np.random.default_rng(freeze['seeds']['composition'])
    for i,cell in enumerate(split_cells()['held_composition']):
        scene=sample_scene(cell,rng,f'reserved-composition-{i:02d}')
        rows.append(dict(role='reserved_composition',scene=scene))
    rng=np.random.default_rng(freeze['seeds']['geometry'])
    for g,geometry in enumerate(NOVEL_GEOMETRIES):
        for draw in range(3):
            # Public factors are reused; only geometry has genuinely new values.
            cell=geometry_reference_cells()[g*3+draw]
            scene=sample_scene(cell,rng,f'reserved-geometry-{g}-{draw}',geometry=geometry)
            rows.append(dict(role='reserved_geometry',scene=scene))
    for row in rows:
        row['actions']=[dict(name=name,pose=pose.tolist()) for name,pose in held_commands(GeometrySpec(**row['scene']['geometry'])).items()]
    return rows


def paired_object_bootstrap(model_errors,baseline_errors,*,seed=926541,resamples=10000):
    """Resample OBJECTS, after aggregating their commands; never pseudo-replicate."""
    model=np.asarray(model_errors,float);baseline=np.asarray(baseline_errors,float)
    if model.ndim!=1 or model.shape!=baseline.shape or len(model)<2 or not np.isfinite(model).all() or not np.isfinite(baseline).all():raise ValueError('Paired finite per-object errors required; failures need a separate count')
    if (model<0).any() or (baseline<0).any():raise ValueError('Nonnegative prediction errors required')
    rng=np.random.default_rng(seed);indices=rng.integers(len(model),size=(resamples,len(model)))
    differences=(baseline-model)[indices].mean(1)
    return dict(objects=len(model),mean_error_reduction=float((baseline-model).mean()),
                percentile95_interval=np.quantile(differences,[.025,.975]).tolist(),
                wins=int((model<baseline).sum()),ties=int((model==baseline).sum()),losses=int((model>baseline).sum()),
                bootstrap_unit='object',seed=seed,resamples=resamples)
