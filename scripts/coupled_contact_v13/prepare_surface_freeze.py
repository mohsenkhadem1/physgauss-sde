"""Two-stage preparation/finalization AFTER an explicit campaign-lead decision.

Preparation does not create method-freeze.json. Finalization requires an
independent exact-artifact audit. This helper never launches simulations.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,platform,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/coupled-surface-v13'
PRIMARY=['slide','curved','moving_sphere'];STRESS=['edge_drop']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def relative(path):
    rel=Path(path)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Use repository-relative input paths')
    return ROOT/rel


def preserved():
    manifest=read(REPORT/'prior-preservation.json')
    if len(manifest['sha256'])!=405:raise RuntimeError('Expected original405 preservation entries')
    for path,sha in manifest['sha256'].items():
        if digest(ROOT/path)!=sha:raise RuntimeError('Prior file changed: '+path)


def no_fresh_reserved(p):
    """No reserved reference/forecast/selection artifacts before prospective freeze."""
    base=ROOT/'runs/coupled-surface-v13'
    ids=[o['id'] for o in p['objects'] if o['split']=='reserved']
    roots=['reference','reference-refinement','reference-spatial','prediction','evaluation','inference','ablation/prediction']
    for directory in roots:
        for object_id in ids:
            path=base/directory/object_id
            if path.exists() and any(item.is_file() for item in path.rglob('*')):
                raise RuntimeError('Fresh reserved artifacts exist before freeze: '+str(path))
    for name in ['reserved-results.json','ablation-results.json']:
        if (REPORT/name).exists():raise RuntimeError('Fresh reserved result report exists before freeze: '+name)


def make_protocol(template,decision):
    if decision.get('status')!='adopted' or decision.get('adopted_scope')!='primary_with_stress' or decision.get('primary_forecast_scenarios')!=PRIMARY or decision.get('stress_forecast_scenarios')!=STRESS:raise RuntimeError('Explicit matching three-primary sharp-stress adoption required')
    p=dict(template)
    p.update(status='Prospectively adopted scope; execution additionally requires independently audited method-freeze.json',adopted_scope='primary_with_stress',primary_forecast_scenarios=PRIMARY,stress_forecast_scenarios=STRESS,
             coarse_cells=[4]*3,reference_cells=[8]*3,dt_coarse=12.5e-6,dt_reference=25e-6,mass_discretization='consistent tetrahedral P1',rigid_inertia='exact uniform-volume P1 moments',surface_spacing_m=.0015,sphere_surface_spacing_m=.0015,friction_quadrature_level=1,
             reference_settings_by_scenario={s:dict(cells=[12 if s=='moving_sphere' else 8]*3,companion_cells=[12 if s=='edge_drop' else 10]*3,dt=12.5e-6 if s=='edge_drop' else 25e-6) for s in ['squeeze','slide','curved','moving_sphere','edge_drop']})
    if set(p['forecast_scenarios'])!={'slide','curved','moving_sphere','edge_drop'}:raise RuntimeError('Pending template scenario matrix changed')
    p['scope']='Fixed three-family primary study (slide, curved, moving_sphere) with original sharp edge_drop retained as development-unqualified exploratory stress. All40forecasts/32reserved retained; primary30overall/24reserved fixed before fresh outcomes. No full PhysTwin or public-dataset improvement claim.'
    from scripts.coupled_contact_v13.engineered_scenarios import build
    from scripts.coupled_contact_v13.inference import observation_nodes
    rest=build('squeeze',cells=(4,4,4)).rest;ids=observation_nodes(rest,p['observation_nodes'])
    p['observation_node_indices']=ids.tolist();p['observation_node_coordinates_m']=rest[ids].tolist()
    return p


def prepare(decision_name,tests_name):
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Already frozen; no overwrite')
    if (REPORT/'method-freeze-candidate.json').exists():raise RuntimeError('Candidate already exists; inspect before any replacement')
    preserved();no_fresh_reserved(read(REPORT/'protocol-pending.json'));decision_path=relative(decision_name);tests=relative(tests_name)
    if not tests.read_text().strip():raise RuntimeError('Test transcript empty')
    p=make_protocol(read(REPORT/'protocol-pending.json'),read(decision_path))
    source_qualification=read(REPORT/'physics-development-qualification.json')
    for name,sha in source_qualification['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Qualified source changed: '+name)
    snapshot=REPORT/'physics-development-qualification-unadopted.json'
    if snapshot.exists():raise RuntimeError('Unadopted snapshot already exists')
    write(snapshot,source_qualification)
    q=dict(source_qualification);q.update(adopted_scope='primary_with_stress',primary_forecast_scenarios=PRIMARY,stress_forecast_scenarios=STRESS,primary_gates_pass=True,all_gates_pass=False,decision=dict(path=str(decision_path.relative_to(ROOT)),sha256=digest(decision_path)),primary_evidence={s:[dict(path=str(snapshot.relative_to(ROOT)),sha256=digest(snapshot),passed_key=['scenarios',s,'qualified'])] for s in ['squeeze']+PRIMARY})
    from scripts.coupled_contact_v13.surface_scope import validate_adoption
    validate_adoption(p,q,ROOT)
    from scripts.coupled_contact_v13.clipped_surface_contact import load_kernel
    _,compiler=load_kernel()
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','warp-lang','torch','matplotlib']}
    inventory=read(REPORT/'runtime-freeze-source-list.json')
    sources={name for group in inventory['groups'].values() for name in group}
    sources.update(source_qualification['source_sha256'])
    sources.update('scripts/coupled_contact_v13/'+name for name in ['surface_scope.py','surface_scope_audit.py','prepare_surface_freeze.py'])
    sources.update(str(path.relative_to(ROOT)) for pattern in ['test_surface*_v13.py','test_surface_pipeline.py','test_surface_supervisor.py','test_surface_artifact_audit.py','test_surface_spatial_quality.py','test_consistent_mass_v13.py','test_coupled_contact_v13_*.py'] for path in (ROOT/'tests').glob(pattern))
    supplement=read(REPORT/'supplementary-protocol.json');supplement['status']='Adopted matching scene/mass/grid supplement; requires independently audited runtime freeze';supplement['source_sha256']={name:digest(ROOT/name) for name in sorted(sources)}
    write(REPORT/'protocol.json',p);write(REPORT/'physics-development-qualification.json',q);write(REPORT/'supplementary-protocol.json',supplement)
    transcript=REPORT/'pre-freeze-tests.txt'
    if tests.resolve()!=transcript.resolve():transcript.write_bytes(tests.read_bytes())
    source_hashes={name:digest(ROOT/name) for name in sorted(sources)}
    candidate=dict(status='CANDIDATE ONLY; independent audit required, not runtime authorization',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=source_hashes,protocol_sha256=digest(REPORT/'protocol.json'),physics_qualification_sha256=digest(REPORT/'physics-development-qualification.json'),tests_sha256=digest(transcript),supplementary_protocol_sha256=digest(REPORT/'supplementary-protocol.json'),prior_preservation_sha256=digest(REPORT/'prior-preservation.json'),decision_sha256=digest(decision_path),python=platform.python_version(),package_versions=versions,compiler=compiler,deadline_utc=read(ROOT/'configs/coupled_contact_v13.json')['deadline_utc'])
    if '05:01:31' not in candidate['deadline_utc']:raise RuntimeError('Original deadline changed')
    preserved();write(REPORT/'method-freeze-candidate.json',candidate)
    print('Candidate prepared; independent pre-freeze audit required. No method-freeze.json or simulations created.')


def finalize(audit_name):
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Already frozen')
    preserved();no_fresh_reserved(read(REPORT/'protocol.json'));candidate_path=REPORT/'method-freeze-candidate.json';c=read(candidate_path);audit_path=relative(audit_name);a=read(audit_path)
    if a.get('passed') is not True or a.get('errors')!=[]:raise RuntimeError('Independent audit did not pass')
    if a.get('candidate_freeze_sha256')!=digest(candidate_path):raise RuntimeError('Audit candidate hash mismatch')
    for field in ['source_sha256','protocol_sha256','physics_qualification_sha256','tests_sha256']:
        if a.get(field)!=c[field]:raise RuntimeError('Audit does not cover exact '+field)
    for name,sha in c['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Source changed after audit: '+name)
    for name,key in [('protocol.json','protocol_sha256'),('physics-development-qualification.json','physics_qualification_sha256'),('pre-freeze-tests.txt','tests_sha256'),('supplementary-protocol.json','supplementary_protocol_sha256'),('prior-preservation.json','prior_preservation_sha256')]:
        if digest(REPORT/name)!=c[key]:raise RuntimeError('Candidate artifact changed: '+name)
    from scripts.coupled_contact_v13.surface_scope import validate_adoption
    validate_adoption(read(REPORT/'protocol.json'),read(REPORT/'physics-development-qualification.json'),ROOT)
    c.update(status='Frozen after independent pre-freeze audit',independent_prefreeze_audit=dict(path=str(audit_path.relative_to(ROOT)),sha256=digest(audit_path)))
    write(REPORT/'method-freeze.json',c)
    print('Runtime freeze installed. No simulation launched.')


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--finalize',action='store_true')
    parser.add_argument('--decision');parser.add_argument('--tests');parser.add_argument('--audit');a=parser.parse_args()
    if a.prepare:
        if not a.decision or not a.tests:parser.error('--prepare requires --decision and --tests')
        prepare(a.decision,a.tests)
    else:
        if not a.audit:parser.error('--finalize requires --audit')
        finalize(a.audit)
if __name__=='__main__':main()
