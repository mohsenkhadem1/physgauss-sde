"""Prospective development-only feasibility of unchanged high native grid scales."""
from pathlib import Path
import concurrent.futures,hashlib,json,multiprocessing,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.native_consistent_development import job,REPORT
from scripts.coupled_contact_v13.campaign import write


def main():
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Must precede fresh method freeze')
    pending=REPORT/'protocol-pending.json';p=json.loads(pending.read_text())
    scales=p['native_scale_grid'][-3:];mus=[0.,.2,.4,.8];tag='consistent-engineered-highgrid-calibration'
    if any(mu not in p['friction_grid'] for mu in mus):raise ValueError('Diagnostic must remain within declared friction grid')
    specs=[dict(cells=4,dt=12.5e-6,friction_level=1,sphere_spacing=.0015,friction=mu,scale=scale,scene='squeeze',tag=tag,name=f'squeeze-{scale:.12g}-mu{mu}') for scale in scales for mu in mus]
    old=json.loads((REPORT/'native-timestep-consistent-engineered-candidate4-declaration.json').read_text())
    sources=list(old['source_sha256'])+[str(Path(__file__).relative_to(ROOT))]
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    plan=dict(scope=__doc__,jobs=specs,source_sha256=hashes,pending_protocol_sha256=hashlib.sha256(pending.read_bytes()).hexdigest(),policy='No changes to declared35point native grid. Same complete squeeze trajectory, mass/contact/integration andpositive-J safety guard. Success establishes nonzero library feasibility, not comparative performance.')
    write(REPORT/f'native-calibration-feasibility-plan.json',plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(job,specs))
    write(REPORT/'native-calibration-feasibility.json',dict(plan=plan,rows=rows,complete=sum(r['success'] for r in rows),expected=len(rows),source_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())))
    print(json.dumps([dict(scale=r['spec']['scale'],mu=r['spec']['friction'],success=r['success'],last_time=r['last_successful_time_s']) for r in rows]))
if __name__=='__main__':main()
