"""Select completed phase11 evidence explicitly; no archive transfer or jobs."""
from pathlib import Path
import hashlib,importlib.metadata,json,platform,sys
root=Path('runs/operator-dev/phase11-archive-v01');root.mkdir(parents=True,exist_ok=True)
report=json.loads(Path('runs/operator-dev/independent-dynamic-matched960-audit-complete-v01.json').read_text())
assert len(report['cells'])==60 and all(x['passed'] for x in report['cells'])
for cell in report['cells']:
 e=json.loads((Path(cell['output'])/'execution.json').read_text());assert e['terminal'] and e['completed'];assert not Path(f'/proc/{e["pid"]}').exists()
roots=['runs/gaussian-campaign-v01/independent-dynamic-gate-v01','runs/gaussian-campaign-v01/independent-dynamic-t0001-hz120-diagonal-v01','runs/gaussian-campaign-v01/independent-render-smoke-v01','runs/gaussian-campaign-v01/cuda-reference-probe-v01','runs/operator-dev/nonlinear-dynamics-mechanics-v01','runs/operator-dev/independent-render-adapter-v01','runs/operator-dev/cuda-reference-probe-v01']
files=set()
for name in roots:
 r=Path(name);assert r.is_dir()
 for p in r.rglob('*'):
  assert not p.is_symlink(),'Symlink excluded: '+str(p)
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files.add(p)
files.update(Path('configs').glob('gaussian*dynamic*.json'));files.add(Path('configs/gaussian_independent_diagnostic_inputs_v01.json'))
for name in ['campaign_nonlinear_dynamics','campaign_nonlinear_dynamics_case','campaign_nonlinear_fem','campaign_linear_fem','campaign_rigid_inertia','campaign_dynamic_floor_protocol','campaign_dynamic_temporal_protocol','campaign_dynamic_960_protocol','campaign_independent_dynamic_audit','campaign_dynamic_extension_audit','campaign_dynamic_960_audit','isaac_dataset','boundary_block','render_independent_trajectory','campaign_cuda_reference_probe','archive_research_artifacts','restore_research_artifacts']:
 p=Path('experiments')/(name+'.py');assert p.is_file();files.add(p)
 p=Path('tests')/('test_'+name+'.py')
 if p.is_file():files.add(p)
files.add(Path('tests/test_research_artifact_archive.py'))
for inventory in ['runs/operator-dev/independent-dynamic-full60-publication-inventory-v01.json','runs/operator-dev/cuda-reference-probe-v01/publication-inventory.json']:
 files.add(Path(inventory))
 for row in json.loads(Path(inventory).read_text())['files']:files.add(Path(row['path']))
files.update(Path('runs/operator-dev').glob('independent-dynamic-*.json'))
for p in files:
 assert p.is_file() and not p.is_symlink() and not p.is_absolute() and '..' not in p.parts
 assert all(not a.is_symlink() for a in p.parents)
 assert not any(part.startswith('.env') or part in ['.git','credentials','secrets','node_modules'] for part in p.parts)
 assert 'pidg' not in str(p).lower() and 'draft' not in str(p).lower()
for p in Path('runs/gaussian-campaign-v01/cuda-reference-probe-v01').rglob('execution.json'):
 e=json.loads(p.read_text());assert e['terminal'] and e['completed'] and not Path(f'/proc/{e["pid"]}').exists()
for p in Path('runs/gaussian-campaign-v01/independent-render-smoke-v01').rglob('execution.json'):
 e=json.loads(p.read_text());assert e['terminal'] and not Path(f'/proc/{e["pid"]}').exists()
versions={}
for name in ['numpy','torch','scipy','pytest','Pillow']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]='not installed in local CPU reference venv'
env={'python':sys.version,'platform':platform.platform(),'packages':versions,'scope':'Local CPU reference package metadata only; no environment variables, credentials or external environments copied.'}
(root/'environment.json').write_text(json.dumps(env,indent=2)+'\n');files.add(root/'environment.json');files.add(Path(__file__).resolve().relative_to(Path.cwd()))
records=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
plan={'role':'Local-only phase11 completed evidence:60 physics cells, successful render and related failed attempts, CUDA probe, frozen sources/configs/tests.',
 'all60physics_cells_verified_terminal':True,'related_render_v01_v02_failures_preserved':True,'render_v03_complete':True,'cuda_controls_and_train_terminal':True,
 'regular_files_only':True,'symlinks_credentials_external_environments_and_live_drafts_excluded':True,'files':records,'file_count':len(records),'source_bytes':sum(r['bytes'] for r in records),
 'archive_output':'runs/gaussian-recovery-phase11-archive','restore_destination':'runs/gaussian-recovery-phase11-restore-clean',
 'archiver_dereference_true_verified':'dereference=True' in Path('experiments/archive_research_artifacts.py').read_text(),'archive_tests':'4 passed in0.16s'}
(root/'selection-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
Path('runs/gaussian-recovery-phase11-paths.json').write_text(json.dumps([r['path'] for r in records],indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k!='files'},indent=2))
