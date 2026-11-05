"""Freeze the twelve-case validation recipe using code and file metadata only."""
import ast,hashlib,json
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import _softvt_path,SOFTVT
from scripts.gaussian_boundary_flow_v02.softvt_validation_inputs import CASES

OUT=Path('reports/gaussian-boundary-flow-v02/softvt-validation')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dependencies(entries):
    pending=[Path(x)for x in entries];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        if not path.is_file():raise ValueError('Missing recipe source: '+str(path))
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom)and node.module:
                modules=[node.module]+[node.module+'.'+n.name for n in node.names]
            for module in modules:
                if not module.startswith(('scripts.','src.')):continue
                candidate=Path(*module.split('.')).with_suffix('.py')
                if candidate.is_file()and candidate not in seen:pending.append(candidate)
    return {str(p):sha(p)for p in sorted(seen)}

def main():
    if (OUT/'protocol.json').exists()or(OUT/'recipe.json').exists():raise FileExistsError('Validation recipe/protocol already frozen; do not overwrite')
    entries=[f'scripts/gaussian_boundary_flow_v02/{name}.py'for name in ('freeze_validation','package_campaign_v04','softvt_validation_inputs','softvt_validation_score','validation_cohort','validation_baseline','bank_retrieval','pose_shape_contact_campaign','retrieval_contact_campaign','prediction_sanity')]
    sources=dependencies(entries)
    cases=[dict(case=f'{suite}-task{task}-demo{episode}',subset=suite,task=task,episode=episode,role='validation')for suite,task,episode in CASES]
    recipe={'status':'frozen','scope':'Prospective validation of unchanged v03 numerical recipe on recurring SoftVT assets; not new-object OOD or final confirmation',
       'cases':[r['case']for r in cases],'geometry':{'type':'convex','nodes':64,'failure_policy':'Record failed geometry; no silent resolution/episode substitution'},
       'inference':{'accepted_bank_worlds':128,'maximum_bank_attempts':256,'training_steps':5000,'posterior_draws':32,'prefix_fraction':.6,'selected_frames':12,'additional_prefix_input':'Inherited gripper/object pose initialization uses the prefix grasp frame and the same48observed identities; not12totalobjectframes','observed_points':48,'noise_m':.001,'numerical_parent':'package_campaign_v03.py','parent_sha256':sha('scripts/gaussian_boundary_flow_v02/package_campaign_v03.py')},
       'methods':['flow_raw','mixture_raw','flow_tracking','mixture_tracking','flow_pose_contact','mixture_pose_contact','nearest_bank_raw','nearest_bank_pose_contact','prefix_persistence','prefix_velocity'],
       'phases':'All twelve prediction pipelines complete or explicitly failed before any validation suffix read; then independent scoring',
       'frozen_failures':'No automatic retries, replacement cases or outcome-driven changes; standalone baseline forecasts survive physics failures',
       'analysis':'Equal-case means and per-cell rows, failure completion first; no pooled-frame significance. Report same-contact retrieval and both simple baselines, shape and raw physical sanity separately.',
       'physical_sanity':['initial raw displacement mismatch','raw floor penetration','pose/contact correction magnitude','surviving draw counts'],
       'limitations':['Relative-feature cancellation remains an identified weakness under test','Quasistatic rest-shape approximation and survivor-conditional draws','Rigid floor projection is not coupled physical inference','No calibrated parameter posterior, rendering improvement or SOTA claim'],
       'protected_roles':'Episodes52-59 not selected; calibration60-69 and confirmation70-99 remain unread',
       'environments':{'export_and_score':'runs/phystwin-env/bin/python','inference':'.venv/bin/python'},'source_sha256':sources}
    source_files={}
    for suite in ('object-soft','spatial-soft'):
        for task in range(3):
            source=_softvt_path(suite,task,SOFTVT).resolve();st=source.stat()
            source_files[f'{suite}/task{task}']={'path':str(source),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    OUT.mkdir(parents=True,exist_ok=True);recipe_path=OUT/'recipe.json'
    recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    sources=dict(sources,**{str(recipe_path):sha(recipe_path)})
    protocol={'schema':1,'status':'frozen','role':'validation','cases':cases,'recipe_path':str(recipe_path.resolve()),'recipe_sha256':sha(recipe_path),
              'source_sha256':sources,'source_files':source_files,'access_ledger':str((OUT/'access-ledger.jsonl').resolve())}
    path=OUT/'protocol.json';path.write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/'protocol.sha256').write_text(sha(path)+'\n')
    print(json.dumps({'protocol':str(path),'sha256':sha(path),'cases':len(cases),'sources':len(sources),'fresh_arrays_read':False}))

if __name__=='__main__':main()
