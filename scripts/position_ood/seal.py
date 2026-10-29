"""Freeze the implementation and learned baseline before held-out inference."""
from pathlib import Path
import datetime,hashlib,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.runtime import ROOT,BASE,CONFIG,BOUNDARY
SOURCES=['active_assist.py','evaluation_cpu.py','predict_mpm_transfer.py','pilot.py','active_sequence.py','active_scoring_wait.py','launch_frozen.py','runtime.py','smooth_fixture.py','differentiable.py','extrapolated.py','fixture.py','material_field.py','inverse.py','run_inverse.py','observations.py','uncertainty.py','adapt.py','phystwin.py','physgaussian.py','fit_mpm.py','neural.py','study.py','score.py','reference_solver.py','predict_transfer.py','transfer.py','calibration.py','active_ablation.py','active_calibration.py','active_jobs.py','campaign_jobs.py','predict_selected.py','neural_lowrank.py','aggregate.py','predict_phystwin.py','seal.py']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    path=BASE/'method-seal.json'
    if not path.exists():raise RuntimeError('Methods must be frozen before held-out inference')
    seal=json.loads(path.read_text())
    for name,expected in seal['hashes'].items():
        p=ROOT/name
        if not p.exists() or digest(p)!=expected:raise RuntimeError(f'Frozen method changed: {name}')
    return seal

def freeze(settings):
    paths=[ROOT/'scripts/position_ood'/s for s in SOURCES]+[CONFIG,BASE/'cohorts.json',BASE/'transfer-protocol.json']
    paths += [ROOT/'scripts/phystwin'/p for p in ['tensor_fixture.py','sequential_fixture.py','v08_reference.py','consistent_fixture.py','gpu_multigrid.py','v06_camera.py','v08_physbridge.py']]
    paths += sorted((ROOT/'runs/third_party/PhysTwin-v06/qqtt').rglob('*.py'))
    paths += sorted((ROOT/'runs/third_party/PhysTwin-v06/configs').glob('*.yaml'))
    paths += list((ROOT/'runs/third_party/PhysGaussian-v09/mpm_solver_warp').glob('*.py'))
    paths += sorted((BASE/'fields').glob('*.npz')) + sorted((BASE/'qualification').glob('*.json'))
    paths += sorted((BASE/'neural-bank/extrapolated-n3').glob('*.npz'))
    paths += [BASE/f'neural/{mode}-{tag}/best.pt' for mode in ['id','all'] for tag in (['n2','extrapolated-n3'] if BOUNDARY=='sharp' else ['extrapolated-n3'])]
    paths += [BASE/f'neural/lowrank-{mode}-extrapolated-n3/best.pt' for mode in ['id','all']] + [BASE/'neural-selection.json']
    seal=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,hashes={str(p.relative_to(ROOT)):digest(p) for p in paths})
    text=json.dumps(seal,indent=2)+'\n';fd=os.open(BASE/'method-seal.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as f:f.write(text)
    return verify()
