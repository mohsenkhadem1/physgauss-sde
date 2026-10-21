"""Batch guard/controller checks without constructing or stepping any physics."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import campaign_hypothesis_batch as batch
from experiments import campaign_hypothesis_dynamics as worker


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(batch,'ROOT',tmp_path)
    sources={}
    for module,name in [(worker,'campaign_hypothesis_dynamics.py'),(batch,'campaign_hypothesis_batch.py')]:
        p=tmp_path/'experiments'/name;p.parent.mkdir(exist_ok=True);p.write_text('# offline guard fixture\n')
        monkeypatch.setattr(module,'SOURCE',p);sources['experiments/'+name]=worker.sha(p)
    monkeypatch.setattr(worker,'runtime_sources',lambda:dict(sources))
    hardware=[dict(id='H'+str(h)) for h in range(4)];hyps=[];cells=[]
    for h in range(4):
        for m in range(3):
            for bc in range(4):
                id=f'H{h}__m{m}__bc{bc}';hyps.append(dict(id=id,public_hardware_id=f'H{h}',spec={'retained_fixture':id}))
                for pi,program in enumerate(batch.PROGRAM_IDS):
                    role='fit' if m in [0,2] and pi==0 else ('development_force_transfer' if m in [0,2] else 'unseen_middle_material')
                    cells.append(dict(id=id+'__'+program,hypothesis_id=id,public_hardware_id=f'H{h}',program_id=program,split='train',evaluation_role=role,status='DECLARED_NOT_EXECUTED'))
    manifest=dict(schema='finite-hardware-first-hypotheses-TRAIN-v0.1',cells=cells,hypotheses=hyps,public_hardware=hardware,program_ids=list(batch.PROGRAM_IDS))
    path=tmp_path/'manifest.json';batch.dump(path,manifest)
    declared=batch.prepare('manifest.json','runs/batch')
    return declared,batch.read(tmp_path/'runs/batch/result-index.json'),manifest


def test_exact96_canonical_roles_and_two_shared_public_clocks_without_sampling(prepared):
    declaration,index,manifest=prepared
    root=worker.ROOT
    assert len(index['records'])==96 and not index['complete']
    assert [r['id'] for r in index['records']]==[r['id'] for r in manifest['cells']]
    assert {role:sum(r['evaluation_role']==role for r in index['records']) for role in ['fit','development_force_transfer','unseen_middle_material']}==dict(fit=32,development_force_transfer=32,unseen_middle_material=32)
    commands={}
    for row in index['records']:
        protocol=worker.validate_protocol(row['protocol']['path'],row['protocol']['sha256'])
        assert protocol['integration_hz']==120 and protocol['evaluation_role']==row['evaluation_role']
        commands.setdefault(row['program_id'],set()).add(protocol['known_commands']['sha256'])
    assert [len(v) for v in commands.values()]==[1,1]
    with pytest.raises(ValueError,match='Fresh'):
        batch.prepare('manifest.json','runs/batch')


@pytest.mark.parametrize('corruption',['role','rate','output','command','spec','id'])
def test_rehashed_cell_cannot_change_declared_identity_or_information(prepared,corruption):
    declaration,index,manifest=prepared;record=index['records'][0]['protocol'];path=worker.resolve(record['path'])
    p=batch.read(path)
    if corruption=='role':p['evaluation_role']='unseen_middle_material'
    if corruption=='rate':p['integration_hz']=240
    if corruption=='output':p['output']='runs/somewhere_else'
    if corruption=='command':p['known_commands']=batch.read(worker.resolve(index['records'][1]['protocol']['path']))['known_commands']
    if corruption=='spec':p['realized_hypothesis']=batch.read(worker.resolve(index['records'][2]['protocol']['path']))['realized_hypothesis']
    if corruption=='id':p['cell_id']='unknown'
    batch.dump(path,p)
    with pytest.raises(ValueError):worker.validate_protocol(record['path'],worker.sha(path))


def test_failure_stops_new_launches_drains_owned_workers_and_retains96_records(prepared,monkeypatch):
    declaration,index,manifest=prepared;launched=[];finished=set()
    monkeypatch.setattr(batch,'completed_exact',lambda c:c['id'] in finished)
    monkeypatch.setattr(batch,'resource_sample',lambda:dict(logical_cpus=36,load_average=[12.,12.,12.],memory_available_bytes=78*1024**3))
    monkeypatch.setattr(batch.time,'sleep',lambda seconds:None)
    class Process:
        def __init__(self,command,**kwargs):
            self.index=len(launched);self.pid=4000000+self.index
            self.id=Path(command[3]).stem;launched.append(self)
        def poll(self):
            if self.index!=1:finished.add(self.id)
            return 1 if self.index==1 else 0
        def wait(self):return self.poll()
    monkeypatch.setattr(batch.subprocess,'Popen',Process)
    result=batch.run(declaration['path'],declaration['sha256'])
    assert len(launched)==8 and len(result['records'])==96
    assert result['terminal'] and not result['complete'] and result['stopped_on_failure']
    assert sum(c['state']=='failed' for c in result['records'])==1
    assert sum(c['completed'] for c in result['records'])==7
    assert sum(c['state']=='queued' for c in result['records'])==88
    assert result['live_count']==0


def test_resource_changes_only_concurrency_not_cases_or_physics():
    resources=dict(maximum_workers=8,memory_reserve_GiB=8,memory_per_worker_GiB=1)
    sample=dict(logical_cpus=36,load_average=[12.,12.,12.],memory_available_bytes=78*1024**3)
    assert batch.resource_capacity(sample,resources)==8
    sample['load_average'][0]=35;assert batch.resource_capacity(sample,resources)==8
    sample.update(load_average=[12.,12.,12.],memory_available_bytes=11*1024**3)
    assert batch.resource_capacity(sample,resources)==3
    sample['memory_available_bytes']=8*1024**3;assert batch.resource_capacity(sample,resources)==0


def test_incomplete_existing_output_is_not_retried(prepared):
    _,index,_=prepared;cell=index['records'][0];output=worker.resolve(cell['output']);output.mkdir(parents=True)
    with pytest.raises(ValueError,match='incomplete output retained'):
        batch.completed_exact(cell)
