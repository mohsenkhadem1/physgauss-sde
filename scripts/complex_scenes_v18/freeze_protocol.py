"""Freeze reviewed development decisions before creating any held worlds.

Requires a complete development census and explicit terminal control evidence.
This command does not generate private worlds or run test queries.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.scenarios import FAMILIES


def check(protocol,audit,decision):
    if protocol['held_data_generated'] or protocol['status']!='draft_not_frozen':raise ValueError('Prospective draft required')
    if {r['family']for r in protocol['families']}!=set(FAMILIES) or protocol['total_queries']!=48 or protocol['independent_worlds']!=24:raise ValueError('Preserve all declared scenario families and worlds')
    keys={(r['case'],r['variant'],r['method'])for r in audit['rows']}
    expected={(f,v,m)for f in FAMILIES for v in (0,1)for m in protocol['methods']}
    if keys!=expected or len(audit['rows'])!=len(expected):raise ValueError('Incomplete development census')
    terminal={'complete','fit_failed','forecast_failed','unsupported'}
    if any(r['state']not in terminal for r in audit['rows']):raise ValueError('Development jobs still pending')
    for r in audit['rows']:
        supported=r['case']in protocol['methods'][r['method']]['families']
        if (r['state']=='unsupported')==supported:raise ValueError('Method coverage changed')
    if decision.get('status')!='reviewed' or not decision.get('rationale') or set(decision['methods'])!=set(protocol['methods']):raise ValueError('Explicit reviewed method settings required')
    for name,values in decision['methods'].items():
        if not values.get('calibration') or values.get('dt_s',0)<=0:raise ValueError('Explicit calibration and integration policy required for '+name)
    if not decision.get('terminal_controls'):raise ValueError('Terminal development controls required')


def freeze(protocol_path,audit_path,decision_path,output):
    paths=[Path(p).resolve()for p in (protocol_path,audit_path,decision_path)];protocol,audit,decision=[json.loads(p.read_text())for p in paths]
    check(protocol,audit,decision);output=Path(output)
    if output.exists():raise ValueError('Preserve frozen protocol')
    dirty=subprocess.check_output(['git','status','--porcelain','--','scripts/complex_scenes_v18','reports/complex-scenes-v18'],cwd=ROOT).decode().strip()
    if dirty:raise ValueError('Commit the reviewed source and reports before freezing')
    evidence={}
    for item in decision['terminal_controls']:
        path=ROOT/item['path'];record=json.loads(path.read_text())
        if sha(path)!=item['sha256'] or record.get(item['terminal_key'])!=item['terminal_value']:raise ValueError('Control evidence changed or nonterminal')
        evidence[item['path']]=item['sha256']
    source={}
    directories=['scripts/complex_scenes_v18','scripts/compositional_neural_mechanics','scripts/compositional_dynamics_v17','scripts/physical_refinement_v15','scripts/phystwin','runs/complex-scenes-v18/vendor','runs/compositional-dynamics-v17/PhysTwin-author']
    for name in directories:
        for p in sorted((ROOT/name).rglob('*')):
            if p.is_file()and p.suffix in ('.py','.cu','.cpp','.h','.toml')and '.git'not in p.parts:source[str(p.relative_to(ROOT))]=sha(p)
    artifact_names=['runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt','runs/compositional-dynamics-v17/matched-appearance-native-resolution/model/point_cloud/iteration_10000/point_cloud.ply','runs/complex-scenes-v18/vendor/UniPhy/ckpt/pretrained_stress_fproj_models_latent_space/ckpt.pth','runs/complex-scenes-v18/vendor/UniPhy/ckpt/pretrained_stress_fproj_models_latent_space/traj_latent_weights_only.pt']
    artifacts={name:sha(ROOT/name)for name in artifact_names};environments={}
    for name in ['.venv','runs/complex-scenes-v18/baseline-env','runs/compositional-dynamics-v17/render-env']:
        python=ROOT/name/'bin/python';environments[name]=dict(python_version=subprocess.check_output([str(python),'--version']).decode().strip(),packages=json.loads(subprocess.check_output([str(python),'-c','import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata["Name"],d.version) for d in m.distributions())))'],cwd=ROOT)))
    for name,settings in decision['methods'].items():protocol['methods'][name].update(settings)
    protocol['calibration_budget_status']='Frozen by recorded development decision'
    protocol['status']='frozen';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    result=dict(status='frozen',held_data_generated=False,source_commit=commit,protocol=protocol,source_sha256=source,artifact_sha256=artifacts,environments=environments,development_evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},terminal_controls_sha256=evidence)
    write(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--audit',required=True);p.add_argument('--decision',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=freeze(a.protocol,a.audit,a.decision,a.output);print(json.dumps(dict(status=result['status'],source_commit=result['source_commit'],source_files=len(result['source_sha256']))))
