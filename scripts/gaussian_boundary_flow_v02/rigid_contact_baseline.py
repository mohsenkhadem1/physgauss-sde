"""Prospective SECONDARY ablation: frozen rigid initial shape + shared pose/contact.

No FEM solve, bank, parameter inference, posterior or physical-accuracy claim.
Original validation recipe/gates are untouched. Prediction only, never scoring.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.pose_shape_contact import predict, POLICY
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha

REPORT=Path('reports/gaussian-boundary-flow-v02/softvt-validation')
ADDENDUM=REPORT/'rigid-contact-secondary-addendum.json'
RUNROOT=Path('runs/gaussian-boundary-flow-v04-validation')
OUTPUT=Path('runs/gaussian-boundary-flow-v04-rigid-contact-secondary')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def predict_initial_shape(data,plan):
    initial=np.broadcast_to(np.asarray(data['nodes']),(1,len(data['times']),len(data['nodes']),3)).copy()
    result,metadata=predict(data,initial,prefix_frames=plan['prefix_frames'],point_ids=plan['point_ids'])
    return result[0],metadata


def freeze():
    if ADDENDUM.exists():raise FileExistsError('Secondary addendum already frozen')
    protocol_path=REPORT/'protocol.json';protocol=json.loads(protocol_path.read_text())
    ledger=Path(protocol['access_ledger'])
    entries=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    if any('evaluator' in x.get('phase','') for x in entries):raise RuntimeError('Cannot preregister after validation target access')
    if (RUNROOT/'evaluation-gate.json').exists():raise RuntimeError('Primary evaluation admission already began')
    sources=[Path(__file__),Path('scripts/gaussian_boundary_flow_v02/pose_shape_contact.py'),Path('scripts/gaussian_boundary_flow_v02/pose_shape.py'),Path('scripts/contact_benchmark_v12/contact_rollout.py'),Path('scripts/contact_benchmark_v12/scene.py')]
    write(ADDENDUM,dict(status='frozen secondary prospective ablation',primary_protocol_sha256=sha(protocol_path),
           primary_recipe_sha256=protocol['recipe_sha256'],cases=[r['case'] for r in protocol['cases']],
           basis='Historically exposed eight-case shape persistence is strong; not selected using unseen validation outcomes',
           method='rigid_initial_shape_contact',policy=POLICY,
           input='Exactly sealed standalone baseline package, same12fitframes48points plus inherited prefix-grasp initialization',
           computation='One repeated initial-node trajectory through unchanged shared pose_shape_contact.predict; no FEM/parameter inference',
           uncertainty='Deterministic; no fabricated duplicate posterior samples or interval claim',
           failures='All12cases; wait for baseline readiness, preserve errors/blocked cases, no retry/replacement',
           analysis='Secondary comparison versus learned/raw/contact and nearest-bank; no change to primary adoption gates, no per-case winner selection',
           evaluation='Root evaluates separately only after original all12terminal barrier; this script never imports/calls a suffix evaluator',
           source_sha256={str(p):sha(p) for p in sources},preexisting_access_ledger_entries=len(entries),
           validation_target_access_before_addendum=False))


def apply_case(case,addendum):
    source=RUNROOT/case/'baseline';out=OUTPUT/case;status=OUTPUT/(case+'-status.json')
    if status.exists():
        previous=json.loads(status.read_text())
        if previous['state']=='running':
            previous.update(state='interrupted',reason='Prior candidate process stopped; no retry')
            write(status,previous)
        return previous
    if not (source/'complete.json').exists():
        statepath=RUNROOT/case/'prediction-status.json'
        if statepath.exists():
            state=json.loads(statepath.read_text());base=state.get('stages',{}).get('baseline',{})
            if base.get('state') in ('failed','blocked','interrupted'):
                row=dict(case=case,state='blocked',baseline_stage=base);write(status,row);return row
        return dict(case=case,state='waiting_for_sealed_baseline')
    write(status,dict(case=case,state='running',retry_policy='No retry of an interrupted or failed candidate'))
    try:
        plan,seal,complete,arrays=load_sealed(source);package=Path(plan['input_package'])
        if sha(package/'manifest.json')!=plan['input_package_manifest_sha256']:raise ValueError('Changed package manifest')
        data=load_package(package)
        if data.get('data_role')!='prospective_validation' or predictor_digest(data)!=plan['input_digest']:
            raise ValueError('Changed role/predictor')
        if plan['validation_protocol_sha256']!=addendum['primary_protocol_sha256']:
            raise ValueError('Different primary validation protocol')
        if not np.array_equal(arrays['initial_geometry'],np.broadcast_to(data['nodes'],arrays['initial_geometry'].shape)):
            raise ValueError('Standalone initial shape changed')
        out.mkdir(parents=True,exist_ok=False)
        derived=dict(plan,prediction_methods=['rigid_contact'],secondary_addendum_sha256=sha(ADDENDUM),
                     scope='SECONDARY deterministic rigid initial shape + shared pose/contact, no FEM or inference',
                     parent_sha256={n:sha(source/n) for n in ['plan.json','inference-sealed.json','complete.json','prediction.npz']})
        write(out/'plan.json',derived)
        prediction,meta=predict_initial_shape(data,plan)
        np.savez_compressed(out/'prediction.npz',rigid_contact=prediction,
                            **{k:data[k] for k in ['nodes','observation_matrix','observed_initial']})
        write(out/'policy-metadata.json',meta)
        write(out/'inference-sealed.json',dict(plan_sha256=sha(out/'plan.json'),forecast_target_access=False,
                                             train_end=data['train_end'],secondary_addendum_sha256=sha(ADDENDUM)))
        write(out/'complete.json',dict(completed=True,held_target_access=False,fit_end=data['train_end'],
                                      frames=len(data['times']),prediction_sha256=sha(out/'prediction.npz'),
                                      uncertainty='Deterministic rigid baseline, no posterior samples'))
        load_sealed(out);row=dict(case=case,state='complete',prediction_sha256=sha(out/'prediction.npz'))
    except Exception as error:row=dict(case=case,state='failed',error=str(error))
    write(status,row);return row


def apply():
    addendum=json.loads(ADDENDUM.read_text())
    if sha(REPORT/'protocol.json')!=addendum['primary_protocol_sha256']:raise ValueError('Primary protocol changed')
    for source,digest in addendum['source_sha256'].items():
        if sha(source)!=digest:raise ValueError('Frozen secondary source changed: '+source)
    rows=[apply_case(case,addendum) for case in addendum['cases']]
    terminal=all(r['state'] in ('complete','failed','blocked','interrupted') for r in rows)
    result=dict(secondary_addendum_sha256=sha(ADDENDUM),cases=rows,all12_terminal=terminal,
                future_target_access=False,interpretation='Interrupted and failed candidates remain explicit; no retries')
    write(OUTPUT/'summary.json',result)
    return terminal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    elif args.watch:
        while not apply():time.sleep(20)
    else:apply()
