"""Final bounded native training-budget control. No held target or geometry changes."""
import importlib.util,json,time
from pathlib import Path
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,sha,write
PUBLIC=ROOT/'runs/compositional-dynamics-v17/calibration-public'
FIT=ROOT/'runs/compositional-dynamics-v17/native-phystwin-budget-fit'
STAGE=ROOT/'runs/compositional-dynamics-v17/native-phystwin-budget-stage'
PLAN=ROOT/'reports/compositional-dynamics-v17/native-phystwin-budget-plan.json'
SOURCE=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_budget_fit.py'

def prepare():
    if PLAN.exists()or FIT.exists()or STAGE.exists():raise ValueError('Preserve prepared budget control')
    prior=json.loads((ROOT/'reports/compositional-dynamics-v17/native-phystwin-fit-plan.json').read_text());meta=json.loads((PUBLIC/'metadata.json').read_text());sources=dict(prior['source_sha256'])
    for p in [Path(__file__),SOURCE,ROOT/'scripts/compositional_dynamics_v17/native_phystwin_replay.py',ROOT/'scripts/compositional_dynamics_v17/native_phystwin_graph_recovery.py',ROOT/'scripts/compositional_dynamics_v17/native_budget_report.py',ROOT/'tests/test_native_phystwin_budget_v17.py']:sources[str(p.relative_to(ROOT))]=sha(p)
    setup=dict(duration_s=.20,frame_dt_s=.01,floor_z_m=meta['public_floor_z_m'],initial_velocity_m_s=[0,0,0],initial_geometry_same=True,gravity_vector_m_s2=[0,0,-9.8])
    write(PLAN,dict(scope='One final development training-budget control before fresh cohort freeze.125particles, same public observations/known geometry/priors, author source and randomseed unchanged. Existing20+200 and729capacity outputs preserved.',source_sha256=sources,input_sha256=prior['input_sha256'],particles=125,cma_max_generations=100,cma_stopping='Unchanged author CMA defaults may terminate before maximum; actualevaluations recorded.',epochs=1000,seed=42,stage_seconds=3600,physical_gpu=0,KillMode='control-group',calibration_setup=setup,forecast_setup=meta['forecast_public_setup'],capacity_comparison='Global calibration-only choice among125/20+200,729/20+200,125/100+1000 using mean squared visible scalar-coordinate error,frames1..20. No held or logged-training-loss selection.',selection_tie='Exact equality retains earlier original recipe; no more budget tuning after this control.',rationale='Native logged objective still decreases at200epochs; fixed-checkpoint replay objective remains the common selection metric.'))

def guard():
    p=json.loads(PLAN.read_text())
    for group in ['source_sha256','input_sha256']:
        for path,digest in p[group].items():
            if sha(ROOT/path)!=digest:raise ValueError('Budget source/input changed '+path)
    return p

def run():
    p=guard();review=json.loads((ROOT/'reports/compositional-dynamics-v17/native-phystwin-budget-prefit-independent-review.json').read_text())
    if review['passed']is not True or review['plan_sha256']!=sha(PLAN):raise ValueError('Exact prefit approval required')
    STAGE.mkdir(exist_ok=False);write(STAGE/'start.json',dict(plan_sha256=sha(PLAN),started_unix=time.time(),stage_seconds=3600));start=time.monotonic();status=dict(completed=False,plan_sha256=sha(PLAN))
    try:
        spec=importlib.util.spec_from_file_location('_native_budget_fit',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.run('fit')
        replay_source=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_replay.py';spec=importlib.util.spec_from_file_location('_native_budget_replay',replay_source);rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep);rep.SOURCE=SOURCE
        for name,setup in [('calibration',p['calibration_setup']),('forecast',p['forecast_setup'])]:
            if time.monotonic()-start>p['stage_seconds']-30:raise TimeoutError('Shared3600s stage budget')
            rep.replay(FIT,PUBLIC,setup,ROOT/f'runs/compositional-dynamics-v17/native-phystwin-budget-{name}',sha(PLAN))
        guard();status['completed']=True
    except Exception as exc:status['error']=repr(exc);raise
    finally:status['wall_seconds']=time.monotonic()-start;write(STAGE/'complete.json',status)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args();globals()[args.stage]()
