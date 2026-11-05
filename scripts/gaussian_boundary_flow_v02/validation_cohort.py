"""Two-phase, failure-preserving twelve-case validation coordinator.

Prediction never invokes an evaluator. Evaluation requires all twelve complete
terminal status records, including explicit blocked/failed stages.
"""
import argparse
from contextlib import contextmanager
import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scripts.gaussian_boundary_flow_v02.softvt_validation_inputs import CASES,admission,sha
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed
# Explicit local dependency imports for recursive freeze-manifest discovery.
from scripts.gaussian_boundary_flow_v02 import validation_baseline,pose_shape_contact_campaign,retrieval_contact_campaign,bank_retrieval,softvt_validation_score

STAGES=('export','baseline','geometry','physics','learned_contact','retrieval','retrieval_contact')
TERMINAL=('complete','failed','blocked','interrupted')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def process_identity(pid=None,command=None):
    pid=os.getpid()if pid is None else pid
    stat=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':stat[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid_namespace':os.readlink('/proc/self/ns/pid'),'command':command if command is not None else [x.decode(errors='replace')for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')if x]}


def process_state(identity):
    if not isinstance(identity,dict)or not all(k in identity for k in ('pid','start_ticks','boot_id','pid_namespace')):return 'uncertain'
    try:
        if identity['pid_namespace']!=os.readlink('/proc/self/ns/pid'):return 'uncertain'
        if identity['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return 'gone'
        stat=(Path('/proc')/str(identity['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        if stat[19]!=identity['start_ticks']or stat[0]in ('Z','X'):return 'gone'
        return 'live'
    except FileNotFoundError:return 'gone'
    except (OSError,KeyError,IndexError):return 'uncertain'


def require_inactive(record):
    pending=[row for row in record.get('stages',{}).values()if row.get('state')not in TERMINAL and row.get('command')]
    if any(not row.get('child')for row in pending)or(record.get('command')and record.get('state')not in TERMINAL and not record.get('child')):raise RuntimeError('Uncertain launch without child identity; refuse resume without writes')
    identities=[record.get('coordinator')]
    if record.get('child')is not None:identities.append(record['child'])
    identities.extend(row['child']for row in record.get('stages',{}).values()if row.get('state')not in TERMINAL and row.get('child')is not None)
    if any(process_state(identity)!='gone'for identity in identities):raise RuntimeError('Live or uncertain prior coordinator/child; refuse resume without writes')


@contextmanager
def cohort_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a+')as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another cohort phase holds the exclusive coordinator lock')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def launch(command,log_path,on_started,timeout):
    with Path(log_path).open('a')as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            identity=process_identity(child.pid,command)
            on_started(identity)
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();raise
        except BaseException:
            try:
                if child.poll()is None:child.kill()
                child.wait(timeout=30)
            except Exception as error:raise KeyboardInterrupt('Child termination uncertain; leave running status for audit')from error
            raise


def preflight(package):
    data=load_package(package)
    if len(data['nodes'])!=64:raise ValueError('Frozen convex64 geometry requires exactly64nodes; no fallback')
    backend=WideGaussianTableBoundaryBackend(data,'cpu');model=backend.model
    tet=model.elements.detach().cpu().numpy();edges=np.concatenate([tet[:,[0,j]]for j in (1,2,3)])
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(64,64))
    components,_=connected_components(graph,directed=False)
    positive=bool((model.mass>0).all())
    if components!=1 or not positive:raise ValueError('Convex64 initial mesh disconnected or contains nonpositive node masses')
    return {'nodes':64,'tetrahedra':len(tet),'connected_components':int(components),'positive_masses':positive,'geometry':'convex','input_manifest_sha256':sha(Path(package)/'manifest.json')}


def execute_case(case_tuple,device,args):
    suite,task,episode=case_tuple;case=f'{suite}-task{task}-demo{episode}';folder=Path(args.run_root)/case;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'prediction-status.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['protocol_sha256']!=args.protocol_sha256:raise ValueError('Changed case protocol')
        if old.get('terminal'):return old
        require_inactive(old)
        for name in STAGES:
            if old['stages'].get(name,{}).get('state')in TERMINAL:continue
            sealed=folder/name
            try:
                if name in ('export','geometry'):raise ValueError('No prediction completion seal')
                load_sealed(sealed)
                old['stages'][name]={'state':'complete','recovered_verified_seal':True}
            except Exception:
                old['stages'][name]={'state':'interrupted','reason':'Prior processes authoritatively absent/reused; no valid sealed completion; no retry'}
        old['terminal']=True;write(path,old);return old
    state={'case':case,'protocol_sha256':args.protocol_sha256,'terminal':False,'coordinator':process_identity(),'stages':{}};write(path,state)
    package=folder/'inputs';py='.venv/bin/python';io='runs/phystwin-env/bin/python';module='scripts.gaussian_boundary_flow_v02.'
    common=['--protocol',args.protocol,'--protocol-sha256',args.protocol_sha256]
    commands={
      'export':([io,'-m',module+'softvt_validation_inputs','--subset',suite,'--task',str(task),'--episode',str(episode),'--output',str(package),'--nodes','64']+common,[],None),
      'baseline':([py,'-m',module+'validation_baseline','--package',str(package),'--output',str(folder/'baseline')]+common,['export'],folder/'baseline'),
      'physics':([py,'-m',module+'package_campaign_v04','--geometry','convex','--package',str(package),'--output',str(folder/'physics'),'--samples','128','--steps','5000','--draws','32','--device',device],['geometry'],folder/'physics'),
      'learned_contact':([py,'-m',module+'pose_shape_contact_campaign','--source',str(folder/'physics'),'--output',str(folder/'learned_contact')],['physics'],folder/'learned_contact'),
      'retrieval':([py,'-m',module+'bank_retrieval','--source',str(folder/'physics'),'--output',str(folder/'retrieval'),'--device',device],['physics'],folder/'retrieval'),
      'retrieval_contact':([py,'-m',module+'retrieval_contact_campaign','--source',str(folder/'retrieval'),'--output',str(folder/'retrieval_contact')],['retrieval'],folder/'retrieval_contact')}
    for name in STAGES:
        dependencies=['baseline']if name=='geometry'else commands[name][1]
        if any(state['stages'][d]['state']!='complete'for d in dependencies):
            state['stages'][name]={'state':'blocked','dependencies':dependencies};write(path,state);continue
        state['stages'][name]={'state':'running','started_unix':time.time()};write(path,state);start=time.monotonic()
        try:
            if name=='geometry':write(folder/'geometry.json',preflight(package))
            else:
                command,_,run=commands[name]
                state['stages'][name]['command']=command;write(path,state)
                def started(identity):
                    state['stages'][name]['child']=identity;state['stages'][name]['command']=command;write(path,state)
                code=launch(command,folder/(name+'.log'),started,7200)
                if code:raise RuntimeError('Stage exit '+str(code))
                if run is not None:load_sealed(run)
            row={'state':'complete','elapsed_s':time.monotonic()-start}
        except Exception as error:row={'state':'failed','elapsed_s':time.monotonic()-start,'error':str(error)}
        row.update({k:state['stages'][name][k]for k in ('child','command')if k in state['stages'][name]})
        state['stages'][name]=row;write(path,state)
    state['terminal']=True;write(path,state);return state


def terminal_gate(run_root,protocol_sha256):
    rows=[]
    for suite,task,episode in CASES:
        case=f'{suite}-task{task}-demo{episode}';path=Path(run_root)/case/'prediction-status.json'
        if not path.exists():raise PermissionError('All12prediction pipelines must terminate before ANY evaluation: missing '+case)
        row=json.loads(path.read_text())
        if row.get('case')!=case or row.get('protocol_sha256')!=protocol_sha256 or row.get('terminal')is not True or set(row.get('stages',{}))!=set(STAGES)or any(v.get('state')not in TERMINAL for v in row['stages'].values()):raise PermissionError('Incomplete/changed cohort prediction state: '+case)
        rows.append({'case':case,'status_sha256':sha(path),'stages':row['stages']})
    return rows


def verified_evaluation(folder,stage,protocol_sha256):
    path=Path(folder)/(stage+'-scores.json')
    try:
        document=json.loads(path.read_text());plan,seal,complete,_=load_sealed(Path(folder)/stage)
        return (document.get('case')==plan['case']and document.get('validation_protocol_sha256')==protocol_sha256
                and document.get('interval')==[seal['train_end'],complete['frames']]
                and document.get('source_sha256',{}).get('prediction')==sha(Path(folder)/stage/'prediction.npz')
                and document.get('source_sha256',{}).get('plan')==sha(Path(folder)/stage/'plan.json'))
    except (OSError,ValueError,KeyError):return False


def evaluate(args):
    root=Path(args.run_root)
    for status in root.glob('*/*-evaluation-status.json'):
        old=json.loads(status.read_text())
        if old.get('state')not in TERMINAL:require_inactive(old)
    rows=terminal_gate(root,args.protocol_sha256)
    gate={'protocol_sha256':args.protocol_sha256,'all12_terminal_before_any_target_read':True,'cases':rows};gatepath=root/'evaluation-gate.json'
    if gatepath.exists()and json.loads(gatepath.read_text())!=gate:raise ValueError('Prediction cohort changed after evaluation admission')
    write(gatepath,gate)
    for row in rows:
        folder=root/row['case']
        for stage in ('baseline','physics','learned_contact','retrieval','retrieval_contact'):
            status=folder/(stage+'-evaluation-status.json')
            if status.exists():
                old=json.loads(status.read_text())
                if old.get('state')in TERMINAL:
                    if old['state']=='complete'and not verified_evaluation(folder,stage,args.protocol_sha256):raise ValueError('Stored evaluator completion no longer verifies')
                    continue
                require_inactive(old)
                recovered=verified_evaluation(folder,stage,args.protocol_sha256)
                write(status,{**old,'state':'complete'if recovered else'interrupted','recovered_verified_score':recovered,'reason':'Prior evaluator/coordinator authoritatively absent; verified score retained or interruption recorded; no retry'})
                continue
            if row['stages'][stage]['state']!='complete':write(status,{'state':'blocked','prediction_stage':row['stages'][stage]});continue
            command=['runs/phystwin-env/bin/python','-m','scripts.gaussian_boundary_flow_v02.softvt_validation_score','--run',str(folder/stage),'--output',str(folder/(stage+'-scores.json')),'--protocol',args.protocol,'--protocol-sha256',args.protocol_sha256,'--cohort-root',str(root)]
            running={'state':'running','cohort_gate_sha256':sha(gatepath),'coordinator':process_identity(),'command':command}
            write(status,running)
            try:
                def started(identity):
                    running['child']=identity;write(status,running)
                code=launch(command,folder/(stage+'-evaluation.log'),started,600)
                write(status,{**running,'state':'complete'if code==0 else'failed','returncode':code})
            except Exception as error:write(status,{**running,'state':'failed','error':str(error)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['predict','evaluate'],required=True);parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--run-root',required=True);args=parser.parse_args()
    # Metadata/admission checks only; no trajectory data opened until explicit predict launch.
    for case in CASES:admission(*case,protocol_path=args.protocol,protocol_sha256=args.protocol_sha256)
    root=Path(args.run_root);root.mkdir(parents=True,exist_ok=True)
    with cohort_lock(root):
        for path in list(root.glob('*/prediction-status.json'))+list(root.glob('*/*-evaluation-status.json')):
            record=json.loads(path.read_text())
            if (path.name=='prediction-status.json'and not record.get('terminal'))or(path.name!='prediction-status.json'and record.get('state')not in TERMINAL):require_inactive(record)
        plan={'protocol_sha256':args.protocol_sha256,'source_sha256':sha(__file__),'cases':[f'{s}-task{t}-demo{e}'for s,t,e in CASES],'devices':['cuda:0','cuda:1','cuda:2']}
        if(root/'cohort-plan.json').exists()and json.loads((root/'cohort-plan.json').read_text())!=plan:raise ValueError('Changed cohort execution plan')
        write(root/'cohort-plan.json',plan)
        if args.phase=='evaluate':evaluate(args);return
        if(root/'evaluation-gate.json').exists():raise ValueError('Cannot resume prediction after evaluation started')
        def worker(i):
            for case in CASES[i::3]:execute_case(case,'cuda:'+str(i),args)
        with ThreadPoolExecutor(max_workers=3)as pool:list(pool.map(worker,range(3)))
        rows=terminal_gate(root,args.protocol_sha256);write(root/'prediction-complete.json',{'all12_terminal':True,'protocol_sha256':args.protocol_sha256,'cases':rows,'evaluation_started':False})

if __name__=='__main__':main()
