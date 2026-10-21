"""Prepare exact published-source inputs for the first authored-force stage.

No simulation is started. The launcher later verifies a fetched source commit
before any child process; this preparation object alone grants no execution.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments import campaign_authored_force_case as runner
from experiments import run_campaign_authored_stage as driver


def record(relative):
    runner.resolve_repo_path(relative)
    return {'path':relative,'sha256':runner.sha(ROOT/relative)}


def prepare():
    expected={
        'experiments/campaign_authored_force_case.py':'97ba4630866772b3399ce68c4dad48d0317486515120de616900784def1af405',
        'experiments/campaign_authored_force_audit.py':'3a91f0a93cb15787db56f6e742afb7047315a96416a35cbf2eeac665f18dab61',
        'experiments/run_campaign_authored_stage.py':'aea6660b3c48aab1ad77abc961bb9dd75e79241f533bdeed2d487a6a930030bf',
        'configs/gaussian_campaign_actual_force_envelope_v01.json':'dd078bd8542cd99015d009e9806e2e09590ee855e6bdb8bafd1cc483ce3d8070',
        'configs/gaussian_campaign_authored_force_cpu_v01.json':'b3da8ba4763f5ec724acf42f4d63bed3edf41d4f679c498ec1d1e595e0339082',
    }
    for relative,wanted in expected.items():
        if runner.sha(ROOT/relative)!=wanted:raise ValueError('Reviewed source changed: '+relative)
    protocol=runner.read(runner.PROTOCOL_PATH)
    stage_record='runs/gaussian-campaign-v01/authored-force-envelope-v01/stage1'
    stage=ROOT/stage_record
    if stage.exists():raise FileExistsError('Preparation already exists; inspect before changing anything')
    manifest=driver.build_initial_manifest(protocol,protocol_sha256=runner.PROTOCOL_SHA256,
        source_sha256=runner.dependencies(),authored_auditor_source_sha256=runner.sha(runner.AUDITOR_PATH),
        output_root=stage_record+'/cells')
    for cell in manifest['cells']:runner.validate_cell(protocol,manifest,cell['id'])
    if any((ROOT/cell['output']).exists() for cell in manifest['cells']):raise FileExistsError('A planned physical output exists')
    stage.mkdir(parents=True,exist_ok=False)
    manifest_record=stage_record+'/resolved-manifest-v01.json'
    driver.write(ROOT/manifest_record,manifest)
    seal={'schema':'actual-authored-force-prelaunch-seal-v0.1','status':'REVIEWED_PUBLISHED_EXECUTION_SEAL',
        'physical_launch_authorized':True,'sources_and_stage_manifest_published':True,
        'lifecycle':'Prepared source object; physical launch is valid only after the outer driver verifies all exact bytes in the explicitly supplied fetched published commit.',
        'protocol':record('configs/gaussian_campaign_actual_force_envelope_v01.json'),
        'resolved_manifest':record(manifest_record),'source_sha256':runner.dependencies(),
        'authored_auditor_source':record('experiments/campaign_authored_force_audit.py'),
        'small_force_complete60_audit':record(protocol['proof_pin_pending_before_launch']['small_force_complete60_audit_path']),
        'resource_declaration':record('configs/gaussian_campaign_authored_force_cpu_v01.json'),
        'stage_driver_source':record('experiments/run_campaign_authored_stage.py')}
    seal_record=stage_record+'/prelaunch-seal-v01.json';driver.write(ROOT/seal_record,seal)
    for cell in manifest['cells']:runner.validate_seal(seal_record,cell['id'])
    preparation={'role':'Offline exact request/source/proof preparation; no physical cell executed',
        'all_51_guard_checks_passed':True,'first_cell':manifest['cells'][0]['id'],
        'first_independent_audit_required':True,'max_parallel_workers_after_first_audit':8,'threads_per_worker':2,
        'expected_reviewed_sha256':expected,'resolved_manifest':record(manifest_record),'prelaunch_seal':record(seal_record),
        'required_published_paths':sorted(set([*runner.validate_seal(seal_record,manifest['cells'][0]['id'])[3],
            'experiments/run_campaign_authored_stage.py'])),
        'launch_command_template':['.venv/bin/python','-u','experiments/run_campaign_authored_stage.py',
            '--prelaunch-seal',seal_record,'--published-commit','<verified-published-source-commit>'],
        'preservation':'No automatic retry/restart, no changed force/mesh/material/core/tolerance, no reserved responses.',
        'production_neural_dataset_admitted':False,'physics_launched':False}
    driver.write(stage/'preparation-v01.json',preparation)
    print(json.dumps({'prepared_cells':len(manifest['cells']),'seal':seal_record,
        'manifest_sha256':runner.sha(ROOT/manifest_record),'seal_sha256':runner.sha(ROOT/seal_record)}))


if __name__=='__main__':prepare()
