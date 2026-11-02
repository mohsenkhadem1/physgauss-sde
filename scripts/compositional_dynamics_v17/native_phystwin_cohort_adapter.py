"""Reusable per-world native recipe. No cohort generation, scoring or automatic launch.
A parent-owned frozen manifest must name public inputs/output directories/queries.
"""
import importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,AUTHOR,sha,write,KEYS
from scripts.compositional_dynamics_v17.native_phystwin_replay import validate_setup,fitted_files,replay
SELECTED_VARIANT='extended125'  # Frozen after final development calibration-only comparison; never per world.
SOURCE=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_budget_fit.py'
RECIPE=dict(cma_generations=100,epochs=1000,cma_stopping='author default early stopping,100is maximum',seed=42,dt_s=.000025,substeps=400,frame_dt_s=.01,calibration_frames=21,tracked_points=98,interior_points=27,initial_graph_radius_m=.04,checkpoint_selection='author prefix-loss best checkpoint',physical_gpu=0)
PUBLIC_METADATA={'train_frame_count','total_command_frames','frame_dt_s','gravity_m_s2','object_masses_kg','total_mass_kg','suffix_targets','scope','public_floor_z_m','gravity_vector_m_s2','initial_velocity_m_s','total_mass_kg_check','native_mass_discretization','material_queries','interior_queries','cameras','visibility_rule','visible_count_per_frame','per_camera_visible_counts','observations_sha256','forecast_public_setup','forecast_public_setups'}

def canonical_points():
    allq=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3)
    surface=np.unique(allq[np.any((allq==0)|(allq==1),axis=1)],axis=0)
    interior=np.stack(np.meshgrid(*([np.array([.25,.5,.75])]*3),indexing='ij'),-1).reshape(-1,3)
    return surface*[.12,.06,.06]+[0,-.03,-.03],interior*[.12,.06,.06]+[0,-.03,-.03]

def validate_public(data,metadata):
    if set(data)!=KEYS or set(metadata)-PUBLIC_METADATA:raise ValueError('Only declared public inputs; no material/family labels')
    shapes=dict(object_points=(21,98,3),object_colors=(21,98,3),object_visibilities=(21,98),object_motions_valid=(20,98),controller_points=(21,0,3),surface_points=(0,3),interior_points=(27,3))
    if any(np.asarray(data[k]).shape!=shape for k,shape in shapes.items()):raise ValueError('Exact21frame/98+27point contract')
    for key in ['object_visibilities','object_motions_valid']:
        if np.asarray(data[key]).dtype!=bool:raise ValueError('Explicit Boolean visibility')
    if not all(np.isfinite(np.asarray(data[k])).all()for k in ['object_points','controller_points','surface_points','interior_points']):raise ValueError('Finite sanitized public geometry/tracks')
    surface,interior=canonical_points()
    if not np.allclose(data['object_points'][0],surface,atol=1e-8,rtol=0)or not np.allclose(data['interior_points'],interior,atol=1e-8,rtol=0):raise ValueError('Known canonical correspondence required')
    visible=data['object_visibilities'];initial=np.broadcast_to(data['object_points'][0],(21,98,3))
    if not np.array_equal(data['object_points'][~visible],initial[~visible]):raise ValueError('Invisible tracks must be initial placeholders')
    if not np.array_equal(data['object_motions_valid'],visible[:-1]&visible[1:]):raise ValueError('Consecutive visibility contract')
    mass=np.asarray(metadata['object_masses_kg'])
    if mass.shape!=(125,)or not np.allclose(mass,.27/125,atol=1e-12,rtol=0):raise ValueError('Same known0.27kg canonical body')
    if metadata['train_frame_count']!=21 or metadata['frame_dt_s']!=.01 or metadata['gravity_vector_m_s2']!=[0,0,-9.8]or metadata['initial_velocity_m_s']!=[0,0,0]:raise ValueError('Fixed calibration clock/initialstate/gravity')


def contained(path):
    p=(ROOT/path).resolve();p.relative_to(ROOT)
    return p

def source_inventory():
    sources=list(AUTHOR.rglob('*.py'))+[AUTHOR/'configs/real.yaml',AUTHOR/'optimize_cma.py',SOURCE]
    sources += [ROOT/name for name in ['scripts/compositional_dynamics_v17/native_phystwin_fit.py','scripts/compositional_dynamics_v17/native_phystwin_inputs.py','scripts/compositional_dynamics_v17/native_phystwin_graph_recovery.py','scripts/compositional_dynamics_v17/native_phystwin_replay.py','scripts/compositional_dynamics_v17/native_phystwin_cohort_adapter.py','tests/test_native_phystwin_cohort_adapter_v17.py']]
    return {str(p.relative_to(ROOT)):sha(p)for p in sources}

def prepare_world(cohort_path,world_name,plan_path):
    cohort_path=contained(cohort_path);plan_path=contained(plan_path);cohort=json.loads(cohort_path.read_text())
    if SELECTED_VARIANT is None or cohort.get('status')!='frozen' or cohort.get('native_recipe')!=RECIPE:raise ValueError('Parent-frozen exact native recipe required')
    rows=[x for x in cohort['native_worlds']if x['name']==world_name]
    if len(rows)!=1:raise ValueError('Unique declared world identity')
    row=rows[0];public=contained(row['public_directory']);output=contained(row['output_directory'])
    if output.exists()or plan_path.exists():raise ValueError('Preserve previous world outputs/plans')
    if not str(output.relative_to(ROOT)).startswith('runs/compositional-dynamics-v17/'):raise ValueError('New v17 output namespace required')
    queries=row['forecast_setups'];names=[q['name']for q in queries]
    if len(names)!=len(set(names))or not names:raise ValueError('Explicit unique query census')
    for q in queries:
        if not q['name'].replace('-','').replace('_','').isalnum():raise ValueError('Safe public query name')
        validate_setup(q['setup'])
    with np.load(public/'observations.npz')as f:data={k:f[k]for k in f.files}
    metadata=json.loads((public/'metadata.json').read_text());validate_public(data,metadata)
    if 'forecast_public_setups'in metadata:
        if any(metadata['forecast_public_setups'].get(q['name'])!=q['setup']for q in queries):raise ValueError('Cohort query differs from declared public setup')
    if metadata['observations_sha256']!=sha(public/'observations.npz'):raise ValueError('Public export integrity')
    inventory=source_inventory()
    for name,digest in inventory.items():
        if cohort['native_source_sha256'].get(name)!=digest:raise ValueError('Native source not frozen before cohort '+name)
    files=[cohort_path,public/'observations.npz',public/'metadata.json',ROOT/'runs/compositional-dynamics-v17/render-environment-freeze.txt']
    write(plan_path,dict(world_name=world_name,recipe=RECIPE,public_directory=str(public.relative_to(ROOT)),output_directory=str(output.relative_to(ROOT)),forecast_setups=queries,source_sha256=inventory,input_sha256={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Fixed native recipe, one calibration fit per world reused across all public drop resets. World IDs route files only, never enter native features or optimization. No evaluator input.',fit_timeout_seconds=3700,replay_timeout_seconds=300))

def guard(plan_path):
    p=json.loads(Path(plan_path).read_text())
    if p['recipe']!=RECIPE:raise ValueError('Native recipe changed')
    for group in ['source_sha256','input_sha256']:
        for name,digest in p[group].items():
            if sha(ROOT/name)!=digest:raise ValueError('World source/input changed '+name)
    return p

def fit_world(plan_path):
    plan_path=contained(plan_path);p=guard(plan_path);spec=importlib.util.spec_from_file_location('_native_world_fit',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if Path(sys.prefix).resolve()!=(ROOT/'runs/compositional-dynamics-v17/render-env').resolve():raise ValueError('Pinned native render-env required')
    m.PLAN=Path(plan_path);m.PUBLIC=contained(p['public_directory']);m.OUT=contained(p['output_directory'])/'fit'
    private_cwd=contained(p['output_directory'])/'native-cwd';private_cwd.mkdir(parents=True,exist_ok=False)
    previous_cwd=Path.cwd()
    try:
        os.chdir(private_cwd);m.run('fit')
    finally:os.chdir(previous_cwd)
    guard(plan_path)

def predict_world(plan_path,query_name):
    plan_path=contained(plan_path);p=guard(plan_path);queries=[x for x in p['forecast_setups']if x['name']==query_name]
    if len(queries)!=1:raise ValueError('Only prospectively declared query')
    if Path(sys.prefix).resolve()!=(ROOT/'runs/compositional-dynamics-v17/render-env').resolve():raise ValueError('Pinned native render-env required')
    base=contained(p['output_directory']);fitted,_=fitted_files(base/'fit')
    if fitted['plan_sha256']!=sha(plan_path):raise ValueError('Per-world fit commitment mismatch')
    replay_source=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_replay.py';spec=importlib.util.spec_from_file_location('_native_selected_replay',replay_source);rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep);rep.SOURCE=SOURCE
    result=rep.replay(base/'fit',contained(p['public_directory']),queries[0]['setup'],base/'forecasts'/query_name,sha(plan_path));guard(plan_path)
    return result

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','fit','predict']);a.add_argument('--plan',type=Path,required=True);a.add_argument('--cohort-plan',type=Path);a.add_argument('--world');a.add_argument('--query');s=a.parse_args()
    if s.stage=='prepare':prepare_world(s.cohort_plan,s.world,s.plan)
    elif s.stage=='fit':fit_world(s.plan)
    else:predict_world(s.plan,s.query)
