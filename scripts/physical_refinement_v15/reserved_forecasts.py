"""Commit sequential beliefs before any actual response scoring.

Imports no truth generator, actual response reader, or response prediction bank.
Only committed measurement manifests reach the selected-only broker.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15.banks import ROOT,assemble,expand,mechanical_design
from scripts.physical_refinement_v15.query_design import prediction_queries
from scripts.physical_refinement_v15.forecast import rollout
from scripts.physical_refinement_v15.measurement_broker import MeasurementBroker,digest
from scripts.physical_refinement_v15.runtime_contract import verify


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def schedules(protocol,camera,track,queries):
    rows=[dict(name='initial_only',policy='fixed',sequence=[],budget=0)]
    rows += [dict(name=f'single-{i:02d}',policy='fixed',sequence=[q.name],budget=1) for i,q in enumerate(queries)]
    rows += [dict(name=f'pair-{i:02d}',policy='fixed',sequence=list(pair),budget=2)
             for i,pair in enumerate(itertools.combinations([q.name for q in queries],2))]
    for row in protocol['fixed_sequences']:
        if row['initial_camera']==camera and row['track']==track:
            rows.append(dict(name='fixed_'+row['category'],policy='fixed',sequence=row['sequence'],budget=2))
    rows += [dict(name=name,policy=name,sequence=[],budget=2) for name in ('visual_uncertainty','physical_risk')]
    if len({r['name'] for r in rows})!=len(rows):raise ValueError('Duplicate policy identity')
    return rows


def validate_packet_manifest(meta,world,repeat,track,queries):
    if (meta['world'],meta['repeat'],meta['track'])!=(world,repeat,track):
        raise ValueError('Measurement identity mismatch')
    dimensions={q.name:q.means.shape[1] for q in queries}
    if set(meta['packets'])!=set(dimensions) or any(meta['packets'][name]['dimension']!=size for name,size in dimensions.items()):
        raise ValueError('Measurement packet allowlist or modality mismatch')


def run(protocol_path,protocol_sha,freeze_path,freeze_sha):
    protocol,freeze=verify(protocol_path,protocol_sha,freeze_path,freeze_sha,artifacts=True)
    base=ROOT/'runs/physical-refinement-v15/reserved'
    out=base/'forecasts'
    if out.exists():raise ValueError('Existing forecast directory; preserve committed experiment')
    packet_root=base/'measurements';generation=packet_root/'complete.json'
    generation_sha=digest(generation)
    packet_commitment=json.loads(generation.read_text())
    if packet_commitment['protocol_sha256']!=protocol_sha or packet_commitment['method_freeze_sha256']!=freeze_sha:
        raise ValueError('Measurement cohort is not from frozen study')
    expected={r['path']:r['sha256'] for w in packet_commitment['worlds'] for r in w['manifests']}
    entries=[r for w in packet_commitment['worlds'] for r in w['manifests']]
    required_paths={f"{w['name']}/repeat-{repeat}/{track}/manifest.json"
                    for w in protocol['reserved_worlds'] for repeat in range(protocol['noise_repeats'])
                    for track in protocol['tracks']}
    if len(entries)!=len(expected) or set(expected)!=required_paths or len(expected)!=96:
        raise ValueError('Exact unique 96 world/repeat/track measurement commitments required')
    expanded=expand(assemble(protocol['gradient_count']),protocol['gradient_count'],protocol['scale_count'])
    design=mechanical_design(expanded)
    rows=[]
    for wi,world in enumerate(protocol['reserved_worlds']):
        for repeat in range(protocol['noise_repeats']):
            for ti,track in enumerate(protocol['tracks']):
                initial,queries=prediction_queries(expanded,world['initial_camera'],track)
                relative=f"{world['name']}/repeat-{repeat}/{track}/manifest.json"
                manifest=packet_root/relative
                meta=json.loads(manifest.read_text())
                validate_packet_manifest(meta,world['name'],repeat,track,[initial]+queries)
                for schedule in schedules(protocol,world['initial_camera'],track,queries):
                    broker=MeasurementBroker(manifest,expected[relative],packet_root=packet_root,budget=schedule['budget'])
                    seed=protocol['policy_seed']+wi*10000+ti*1000+repeat*10
                    weights,info=rollout(expanded['weights'],initial,queries,design,broker.request,
                                        policy=schedule['policy'],fixed_sequence=schedule['sequence'],
                                        seed=seed,fantasies=protocol['fantasies'],budget=schedule['budget'])
                    folder=out/world['name']/f'repeat-{repeat}'/track/schedule['name'];folder.mkdir(parents=True)
                    np.savez_compressed(folder/'weights.npz',weights=weights)
                    record=dict(world=world['name'],initial_camera=world['initial_camera'],repeat=repeat,track=track,
                                policy=schedule['name'],rollout=info,measurement_receipt=broker.receipt(),
                                weights_sha256=digest(folder/'weights.npz'),measurement_manifest_sha256=expected[relative])
                    write(folder/'complete.json',record)
                    rows.append(dict(path=str(folder.relative_to(out)),sha256=digest(folder/'complete.json')))
                print(world['name'],repeat,track,'forecasts committed',flush=True)
    verify(protocol_path,protocol_sha,freeze_path,freeze_sha,artifacts=True)
    if digest(generation)!=generation_sha:raise ValueError('Measurement generation changed')
    write(out/'complete.json',dict(role='all_forecasts_committed_before_response_scoring',
          protocol_sha256=protocol_sha,method_freeze_sha256=freeze_sha,rows=rows,
          measurement_generation_sha256=digest(generation),
          scope='Finite ensemble sequential forecasts, actual held response arrays never read'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    run(a.protocol,a.protocol_sha256,a.freeze,a.freeze_sha256)
