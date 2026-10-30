"""Uniform post-freeze fairness baseline; primary candidate/results are unchanged."""
from pathlib import Path
import argparse,datetime,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,check_deadline,sha256,write_json
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.scene import load_scene
from scripts.contact_benchmark_v12.contact_rollout import rollout
from scripts.external_benchmark.evaluate import score
METHOD='adapted_phystwin_scene_state'
FREEZE=REPORT/'supplemental-baseline-freeze.json'
SOURCES=['supplemental_baselines.py','contact_rollout.py','scene.py','data.py','object_frame.py']

def hashes():return {f'scripts/contact_benchmark_v12/{n}':sha256(ROOT/'scripts/contact_benchmark_v12'/n) for n in SOURCES}
def freeze():
    if not (REPORT/'method-freeze.json').exists():raise RuntimeError('Primary freeze must exist')
    if FREEZE.exists():raise RuntimeError('Supplement freeze already exists; do not replace')
    protocol=json.loads((REPORT/'benchmark-protocol.json').read_text())
    write_json(FREEZE,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='Additional fairness baseline proposed after primary freeze and partial primary scores; no primary selection or tuning. Apply identical frozen static-scene pose postprocessor to adapted native PhysTwin on all80SoftVT sequences.',method=METHOD,cohorts={c:protocol[c]['SoftVTBench'] for c in ['regression','untouched_confirmation']},parameters=dict(detect_detachment=True,friction=.5,substeps=80),source_sha256=hashes(),primary_method_freeze_sha256=sha256(REPORT/'method-freeze.json'),protocol_sha256=sha256(REPORT/'benchmark-protocol.json'),scope='Common position/scene postprocessing, not the authors unchanged PhysTwin method; no refit, no future-error choice.'))

def run(spec,cohort,decision):
    check_deadline();name=f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
    out=BASE/'evaluation'/cohort/name
    dest=out/'native-scene-supplement.json'
    if dest.exists():return True
    source=(ROOT/'runs/external-benchmark-v11/native/primary' if cohort=='regression' else BASE/'native/confirmation')/name/'prediction.npz'
    if not source.exists() or not (source.parent/'fit.json').exists() or not (out/'metrics.json').exists():return False
    data=load(spec);native=np.load(source)
    np.testing.assert_array_equal(native['nodes'],data['nodes'])
    base=native['predicted'];assert base.shape==(len(data['times']),len(data['nodes']),3)
    scene=load_scene(predictor_payload(data))
    pred,meta=rollout(predictor_payload(data),base,scene,**decision['parameters'])
    start=data['train_end'];end=len(data['times'])
    result=score(data,pred,start,end)
    masks={'commanded_hold':data['attachment_active'].sum(1)>0,'commanded_release':(np.arange(end)>data['grasp'])&(data['gripper_command']>0)}
    segments={key:score(data,pred,start,end,mask) for key,mask in masks.items()}
    np.savez_compressed(out/'native-scene-prediction.npz',predicted=pred,nodes=data['nodes'])
    write_json(dest,dict(name=name,dataset=data['dataset'],cohort=cohort,method=METHOD,result=result,segments=segments,metadata=meta,source_native=str(source.relative_to(ROOT)),source_native_sha256=sha256(source),node_identity_verified=True,decision_sha256=sha256(FREEZE),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(cohort,name,'supplement complete',flush=True)
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);a=p.parse_args()
    if a.freeze:freeze();return
    decision=json.loads(FREEZE.read_text())
    assert decision['source_sha256']==hashes(),'Frozen supplement sources changed'
    assert 0<=a.part<a.parts
    jobs=[(c,s) for c,specs in decision['cohorts'].items() for s in specs][a.part::a.parts]
    while True:
        count=0
        for cohort,spec in jobs:count+=run(spec,cohort,decision)
        print('Supplement',count,'/',len(jobs),flush=True)
        if count==len(jobs) or not a.watch:break
        check_deadline();time.sleep(20)
if __name__=='__main__':main()
