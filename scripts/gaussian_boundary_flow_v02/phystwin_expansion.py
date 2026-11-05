"""Frozen four-case retrospective prediction queue with an evaluation barrier."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.gaussian_boundary_flow_v02.validation_cohort import process_identity,require_inactive,cohort_lock,launch,write,TERMINAL
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.gaussian_boundary_flow_v02.freeze_validation import dependencies
from scripts.gaussian_boundary_flow_v02 import phystwin_expansion_baseline,package_campaign_v03,pose_shape_contact_campaign,bank_retrieval,retrieval_contact_campaign

CASES=('double_stretch_sloth','double_stretch_zebra','single_lift_dinosor','double_lift_sloth')
STAGES=('baseline','geometry','physics','learned_contact','retrieval','retrieval_contact')


def freeze(path):
    path=Path(path)
    if path.exists():raise FileExistsError('Expansion plan is immutable')
    geometry_path=Path('reports/gaussian-boundary-flow-v02/phystwin-v05-selected-geometry.json');geometry=json.loads(geometry_path.read_text());lookup={r['case']:r for r in geometry['rows']};jobs=[]
    for i,case in enumerate(CASES):
        row=lookup[case];package=Path(row['package']);data=load_package(package)
        if data['name']!=case or len(data['nodes'])!=row['nodes']or sha(package/'manifest.json')!=row['selected_manifest_sha256']:raise ValueError('Changed admitted input geometry')
        jobs.append({'case':case,'package':str(package.resolve()),'manifest_sha256':row['selected_manifest_sha256'],'input_digest':row['selected_predictor_digest'],'nodes':row['nodes'],'geometry':'filtered','device':'cuda:'+str(i%2)})
    entries=[__file__,'scripts/gaussian_boundary_flow_v02/phystwin_expansion_baseline.py','scripts/gaussian_boundary_flow_v02/package_campaign_v03.py','scripts/gaussian_boundary_flow_v02/object_score.py']
    plan={'status':'frozen','role':'historically_exposed_development','cases':jobs,'source_sha256':dependencies(entries),'selected_geometry_sha256':sha(geometry_path),
          'inference':{'accepted_bank_worlds':128,'maximum_bank_attempts':256,'training_steps':5000,'posterior_draws':32,'prefix_frames':12,'observed_points':48},
          'scope':'Four historically exposed PhysTwin plush cases. Same geometry and prefix budgets within each case; repaired 119-node cases are changed discretizations, not a convergence claim.',
          'failure_policy':'Preserve independent baselines; no automatic retries, case replacement, geometry changes or selection using suffix scores.',
          'barrier':'All four complete prediction pipelines must be terminal before any evaluation. Prediction and evaluation require separate explicit launches.',
          'comparisons':['raw flow and mixture','learned pose/contact','nearest bank raw and same pose/contact','prefix persistence and velocity','initial shape with the same rigid pose/contact policy'],
          'limitations':['Retrospective exposure, not fresh validation or new-object generalization','Survivor-conditional posterior draws and approximate rigid contact','No force or material calibration claim']}
    write(path,plan);return plan


def check_plan(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen expansion plan digest mismatch')
    plan=json.loads(Path(path).read_text())
    if plan['status']!='frozen'or [r['case']for r in plan['cases']]!=list(CASES):raise ValueError('Unexpected expansion cases')
    for source,expected in plan['source_sha256'].items():
        if sha(source)!=expected:raise ValueError('Frozen source changed: '+source)
    for job in plan['cases']:
        if sha(Path(job['package'])/'manifest.json')!=job['manifest_sha256']:raise ValueError('Frozen package changed')
    return plan


def execute(job,args):
    folder=Path(args.run_root)/job['case'];folder.mkdir(parents=True,exist_ok=True);status=folder/'prediction-status.json'
    if status.exists():
        old=json.loads(status.read_text())
        if old['plan_sha256']!=args.plan_sha256:raise ValueError('Changed case plan')
        if old.get('terminal'):return
        require_inactive(old)
        for name in STAGES:
            if old['stages'].get(name,{}).get('state')in TERMINAL:continue
            try:
                if name=='geometry':raise ValueError('No prediction seal')
                load_sealed(folder/name);old['stages'][name]={'state':'complete','recovered_verified_seal':True}
            except Exception:old['stages'][name]={'state':'interrupted','reason':'Prior processes absent; no valid seal and no retry'}
        old['terminal']=True;write(status,old);return
    state={'case':job['case'],'plan_sha256':args.plan_sha256,'coordinator':process_identity(),'terminal':False,'stages':{}};write(status,state)
    py='.venv/bin/python';module='scripts.gaussian_boundary_flow_v02.'
    jobs={
      'baseline':([py,'-m',module+'phystwin_expansion_baseline','--package',job['package'],'--output',str(folder/'baseline')],[]),
      'physics':([py,'-m',module+'package_campaign_v03','--geometry','filtered','--package',job['package'],'--output',str(folder/'physics'),'--samples','128','--steps','5000','--draws','32','--device',job['device']],['geometry']),
      'learned_contact':([py,'-m',module+'pose_shape_contact_campaign','--source',str(folder/'physics'),'--output',str(folder/'learned_contact')],['physics']),
      'retrieval':([py,'-m',module+'bank_retrieval','--source',str(folder/'physics'),'--output',str(folder/'retrieval'),'--device',job['device']],['physics']),
      'retrieval_contact':([py,'-m',module+'retrieval_contact_campaign','--source',str(folder/'retrieval'),'--output',str(folder/'retrieval_contact')],['retrieval'])}
    for name in STAGES:
        needed=['baseline']if name=='geometry'else jobs[name][1]
        if any(state['stages'][key]['state']!='complete'for key in needed):state['stages'][name]={'state':'blocked','dependencies':needed};write(status,state);continue
        state['stages'][name]={'state':'running','started_unix':time.time()};write(status,state)
        try:
            if name=='geometry':
                backend=GeometryAwareGaussianBoundaryBackend(load_package(job['package']),'cpu');write(folder/'geometry.json',backend.geometry_diagnostic)
                if backend.model.n!=job['nodes']:raise ValueError('Geometry node count changed')
            else:
                command=jobs[name][0];state['stages'][name]['command']=command;write(status,state)
                def started(identity):state['stages'][name]['child']=identity;write(status,state)
                code=launch(command,folder/(name+'.log'),started,7200)
                if code:raise RuntimeError('Stage exit '+str(code))
                load_sealed(folder/name)
            state['stages'][name]['state']='complete'
        except Exception as error:state['stages'][name].update(state='failed',error=str(error))
        write(status,state)
    state['terminal']=True;write(status,state)


def terminal_gate(root,digest):
    rows=[]
    for case in CASES:
        path=Path(root)/case/'prediction-status.json'
        if not path.exists():raise PermissionError('All four prediction pipelines must terminate before evaluation')
        row=json.loads(path.read_text())
        if row.get('case')!=case or row.get('plan_sha256')!=digest or row.get('terminal')is not True or set(row.get('stages',{}))!=set(STAGES)or any(r.get('state')not in TERMINAL for r in row['stages'].values()):raise PermissionError('Incomplete or changed prediction cohort')
        rows.append({'case':case,'status_sha256':sha(path),'stages':row['stages']})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','predict'],required=True);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256');p.add_argument('--run-root',default='runs/gaussian-boundary-flow-v05-phystwin');p.add_argument('--wait-for-device',choices=['cuda:0','cuda:1'],action='append',default=[]);a=p.parse_args()
    if a.phase=='freeze':freeze(a.plan);print(sha(a.plan));return
    plan=check_plan(a.plan,a.plan_sha256);root=Path(a.run_root)
    with cohort_lock(root):
        for path in root.glob('*/prediction-status.json'):
            old=json.loads(path.read_text())
            if not old.get('terminal'):require_inactive(old)
        if(root/'evaluation-gate.json').exists():raise ValueError('Evaluation already admitted; prediction cannot resume')
        execution={'plan_sha256':a.plan_sha256,'run_root':str(root.resolve())}
        if(root/'execution.json').exists()and json.loads((root/'execution.json').read_text())!=execution:raise ValueError('Changed execution plan')
        write(root/'execution.json',execution)
        def worker(device):
            if device in a.wait_for_device:
                permit=root/'device-permits'/(device.replace(':','-')+'.json');deadline=time.monotonic()+7200
                while not permit.exists():
                    if time.monotonic()>deadline:raise TimeoutError('Device release permit not supplied; unstarted cases remain unstarted')
                    time.sleep(10)
                if json.loads(permit.read_text())!={'device':device,'plan_sha256':a.plan_sha256}:raise ValueError('Device permit identity mismatch')
            for job in plan['cases']:
                if job['device']==device:execute(job,a)
        with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(worker,['cuda:0','cuda:1']))
        write(root/'prediction-complete.json',{'all_four_terminal':True,'plan_sha256':a.plan_sha256,'cases':terminal_gate(root,a.plan_sha256),'target_access':False})

if __name__=='__main__':main()
