"""Declared half-step checks of both completed native calibration candidates."""
from pathlib import Path
import concurrent.futures,hashlib,json,multiprocessing,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.native_consistent_development import job,REPORT,BASE
from scripts.coupled_contact_v13.campaign import write


def main():
    if (REPORT/'method-freeze.json').exists():raise RuntimeError('Development must precede freeze')
    path=REPORT/'native-calibration-feasibility.json';source=json.loads(path.read_text());successful=[r for r in source['rows'] if r['success']]
    if len(successful)!=2:raise ValueError('Expected exactly two completed declared candidates')
    specs=[];inputs={}
    for row in successful:
        s=dict(row['spec']);s.update(dt=s['dt']/2,tag='consistent-engineered-calibration-halfstep');specs.append(s)
        for name in ['trajectory.npz','diagnostic.json']:
            p=BASE/row['spec']['tag']/row['spec']['name']/name;inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source['plan']['source_sha256']}
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan=dict(scope=__doc__,jobs=specs,source_sha256=hashes,input_sha256=inputs,feasibility_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gates=dict(trajectory_coordinate_rms_m=.00025,penetration_m=.001,min_j=.02),no_reserved_inputs=True)
    write(REPORT/'native-calibration-temporal-plan.json',plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(job,specs))
    comparisons=[]
    for old,new in zip(successful,rows):
        metric=dict(scale=old['spec']['scale'],friction=old['spec']['friction'],complete=bool(old['success'] and new['success']),passed=False)
        if metric['complete']:
            with np.load(BASE/old['spec']['tag']/old['spec']['name']/'trajectory.npz') as a,np.load(BASE/new['spec']['tag']/new['spec']['name']/'trajectory.npz') as b:
                same=a['frames'].shape==b['frames'].shape and np.allclose(a['times'],b['times'],atol=1e-10,rtol=0)
                finite=np.isfinite(a['frames']).all() and np.isfinite(b['frames']).all()
                if not same or not finite:raise ValueError('Invalid temporal comparison arrays')
                difference=a['frames'][1:]-b['frames'][1:]
                rms=float(np.sqrt(np.mean(difference**2)));final=float(np.sqrt(np.mean(difference[-1]**2)))
            physical=all(np.isfinite(r['max_penetration_m']) and r['max_penetration_m']<=.001 and np.isfinite(r['integrated_min_j']) and r['integrated_min_j']>.02 for r in [old,new])
            metric.update(trajectory_coordinate_rms_m=rms,final_frame_coordinate_rms_m=final,finite=True,physical_gates=physical,passed=bool(physical and rms<=.00025))
        comparisons.append(metric)
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in {**hashes,**inputs}.items())
    write(REPORT/'native-calibration-temporal.json',dict(plan=plan,rows=rows,comparisons=comparisons,source_and_inputs_unchanged=unchanged,all_checks_pass=unchanged and all(c['passed'] for c in comparisons),scope='Two completed candidates from12 tested high-grid hypotheses; not full35-grid census or forecast qualification. Firstframeexcluded from trajectory RMS.'))
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
