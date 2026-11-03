"""Retain terminal prediction slots whose prerequisite fit could not run.

Infrastructure interruptions are explicitly separated from scientific fit and
rollout failures. No held reference trajectory or private material is opened.
"""
import json
from pathlib import Path
from scripts.complex_scenes_v18.held_manifest import sha
from scripts.complex_scenes_v18.run_development import ROOT,write


def failure_kind(task,tasks,states):
    roots=[]
    def walk(t):
        if states[t['id']]['state']=='dependency_failed':
            for dep in t['dependencies']:
                if states[dep]['state']!='complete':walk(tasks[dep])
        else:roots.append(t)
    walk(task)
    for t in roots:
        if t['role']=='reference'and t['id'].endswith(':calibration')and states[t['id']]['state']=='failed':
            args=t['argv'];campaign=Path(args[args.index('--campaign')+1]);world=args[args.index('--world-id')+1]
            records=list((campaign/'references'/world/'calibration').glob('*/complete.json'))
            if len(records)==1 and json.loads(records[0].read_text()).get('completed')is False:return 'calibration_reference_invalid',[t['id']for t in roots]
    scientific=True
    for t in roots:
        if t['role']!='fit' or states[t['id']]['state']!='failed':scientific=False;continue
        args=t['argv'];label=args[args.index('--label')+1]
        # A recorded unsuccessful fit differs from an exception before its seal.
        paths=list((ROOT/'runs/complex-scenes-v18'/label).glob('*/fit-seal.json'))
        if len(paths)!=1 or json.loads(paths[0].read_text()).get('completed')is not False:scientific=False
    return ('fit_failed'if scientific and roots else'execution_failed'),[t['id']for t in roots]


def seal(campaign,plan,states):
    campaign=Path(campaign);tasks={t['id']:t for t in plan['tasks']};written=[]
    terminal={'complete','failed','dependency_failed','interrupted'}
    if any(s['state']not in terminal for s in states.values()):raise ValueError('Only terminal task census can be sealed')
    for task in tasks.values():
        if task['role']!='prediction':continue
        out=campaign/'predictions'/task['method']/task['world_id']/f"query{task['query_index']}";path=out/'prediction-seal.json'
        if path.exists():continue
        if states[task['id']]['state']=='complete':raise ValueError('Successful prediction task omitted its seal')
        kind,causes=failure_kind(task,tasks,states)
        record=dict(completed=False,state=kind,evaluation_split='held',method=task['method'],world_id=task['world_id'],query_index=task['query_index'],public_manifest_sha256=sha(campaign/'public-manifest.json'),forecast_targets_read=False,error='Prediction not produced; inspect causal task records',causal_task_ids=causes,causal_task_states={i:states[i]for i in causes})
        write(path,record);written.append(str(path))
    return written
