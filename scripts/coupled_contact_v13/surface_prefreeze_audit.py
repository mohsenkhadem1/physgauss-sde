"""Read-only pre-freeze inventory and blockers; never authorizes a campaign."""
from pathlib import Path
import argparse,hashlib,json,datetime,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));REPORT=ROOT/'reports/coupled-surface-v13'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',default=str(REPORT/'protocol-pending.json'));args=parser.parse_args()
    p=json.loads(Path(args.protocol).read_text());blockers=[];checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:blockers.append(name)
    prior=json.loads((REPORT/'prior-preservation.json').read_text())['sha256'];changed=[f for f,h in prior.items() if not (ROOT/f).exists() or digest(ROOT/f)!=h]
    check('All405 prior artifacts unchanged',len(prior)==405 and not changed)
    check('Ten unique objects',len(p['objects'])==10 and len({o['id'] for o in p['objects']})==10)
    check('Two development and eight reserved objects',sum(o['split']=='development' for o in p['objects'])==2 and sum(o['split']=='reserved' for o in p['objects'])==8)
    scenes=['squeeze']+p['forecast_scenarios'];check('Five unique scenarios',len(scenes)==5 and len(set(scenes))==5)
    for key in ('coarse_cells','dt_coarse','surface_spacing_m','friction_quadrature_level'):
        check('Resolved '+key,p.get(key) is not None)
    refs=p.get('reference_settings_by_scenario',{})
    for scene in scenes:
        q=refs.get(scene,{});cells=q.get('cells');companion=q.get('companion_cells');dt=q.get('dt')
        valid=isinstance(cells,list) and isinstance(companion,list) and len(cells)==len(companion)==3 and cells!=companion and isinstance(dt,(float,int)) and dt>0
        check(scene+' explicit distinct primary/companion and base dt',valid)
        if valid and p.get('coarse_cells'):
            check(scene+' primary nests common predictor',all(n%c==0 for n,c in zip(cells,p['coarse_cells'])))
    qualification=json.loads((REPORT/'physics-development-qualification.json').read_text())
    from scripts.coupled_contact_v13.surface_scope_audit import validate_scope
    try:
        validate_scope(p,qualification,dict(physics_qualification_sha256=digest(REPORT/'physics-development-qualification.json')),ROOT)
        check('Explicit adopted scope and primary-specific qualified evidence',True)
    except (ValueError,KeyError,OSError) as error:
        check('Explicit adopted scope and primary-specific qualified evidence: '+str(error),False)
    mismatches=[f for f,h in qualification.get('source_sha256',{}).items() if not (ROOT/f).exists() or digest(ROOT/f)!=h]
    check('Development source inventory available and unchanged',bool(qualification.get('source_sha256')) and not mismatches)
    names=['physics.py','pipeline.py','scenarios.py','inference.py','baselines.py','native_adapter.py','rigid.py','surface_contact.py','sparse_surface_contact.py','clipped_surface_contact.py','clipped_surface_contact.cpp','consistent_mass.py','engineered_scenarios.py','surface_pipeline.py','surface_body_quality.py','surface_spatial_quality.py','surface_ablation.py','surface_verify.py','surface_spatial_verify.py','verify.py','verify_surface_ablation.py','surface_render.py','render.py','surface_report.py','surface_run_campaign.py','surface_finish.py','mesh_diagnostic.py','surface_prefreeze_audit.py','surface_scope.py','surface_scope_audit.py','material_recovery_report.py','campaign.py']
    inventory={}
    for name in names:
        f=ROOT/'scripts/coupled_contact_v13'/name;check('Inventory exists '+name,f.exists())
        if f.exists():inventory[str(f.relative_to(ROOT))]=digest(f)
    check('No fresh reserved references before freeze',not any((ROOT/'runs/coupled-surface-v13/reference').glob('surface_test*/**/trajectory.npz')))
    output=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=str(Path(args.protocol).relative_to(ROOT)),protocol_sha256=digest(Path(args.protocol)),automated_checks_pass=not blockers,checks=checks,blockers=blockers,prior_preserved=len(prior)-len(changed),prior_changed=changed,qualification_source_mismatches=mismatches,source_inventory_sha256=inventory,required_final_evidence_review=[
      'Calibration and every declared PRIMARY scenario x both development materials pass spatial sparse+dense+body checks at the actual selected primary/companion and dt; symmetric pair acceptance is sufficient.',
      'Both members of every chosen pair have base/half timestep evidence at production base dt, including dynamic bodies; original lumped or different-dt evidence is not inherited.',
      'Selected PRIMARY configurations pass contact/friction/sphere integration checks and fine-reference endpoint matrix; predictor endpoints and temporal evidence cover the common predictor setting.',
      'Qualification evidence digests resolve through immutable snapshots/ledger when live progress manifests were overwritten; sources and trajectories are preserved.',
      'Main freeze covers inventory plus third-party native source, compiler source/binary/compiler flags provenance, qualification/test reports, supplemental protocol and observation indices/coordinates.',
      'Runtime schedules all150 reference jobs and40forecasts; absent failed/unqualified companions cannot change primary targets or silently omit cases; final verifier checks complete accounting.'
    ],scope='Automated inventory/checklist only. Human/agent evidence coverage review is still required; this file does not authorize freeze or claim convergence.')
    (REPORT/'pre-freeze-independent-checklist.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(dict(automated_checks_pass=not blockers,blockers=blockers,prior_preserved=output['prior_preserved']),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
