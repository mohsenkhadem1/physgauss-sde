"""Freeze the implementation and learned baseline before held-out inference."""
from pathlib import Path
import datetime,hashlib,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/position-ood-v09'
SOURCES=['differentiable.py','fixture.py','material_field.py','inverse.py','run_inverse.py','observations.py','uncertainty.py','adapt.py','phystwin.py','physgaussian.py','fit_mpm.py','neural.py','study.py','seal.py']

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
    paths=[ROOT/'scripts/position_ood'/s for s in SOURCES]+[ROOT/'configs/position_ood_v09.json',BASE/'cohorts.json']
    paths += [ROOT/'scripts/phystwin'/p for p in ['tensor_fixture.py','sequential_fixture.py','v08_reference.py','consistent_fixture.py','gpu_multigrid.py','v06_camera.py']]
    paths += [ROOT/'runs/third_party/PhysTwin-v06/qqtt/model/diff_simulator/spring_mass_warp.py']
    paths += list((ROOT/'runs/third_party/PhysGaussian-v09/mpm_solver_warp').glob('*.py'))
    paths += [BASE/f'neural/{mode}-n2/best.pt' for mode in ['id','all']]
    seal=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,hashes={str(p.relative_to(ROOT)):digest(p) for p in paths})
    text=json.dumps(seal,indent=2)+'\n';fd=os.open(BASE/'method-seal.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as f:f.write(text)
    return verify()
