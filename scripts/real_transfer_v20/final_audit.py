"""Independent arithmetic, frozen-parameter and numerical-validity checks."""
import json,pickle,numpy as np
from .development_run import ROOT,BASE,write,sha
from .registration import PAIRS
from .metrics import tetra_sanity
from .refined_campaign import LABEL
from scripts.real_transfer_v19.phystwin_campaign import CASES
from scripts.real_transfer_v19.phystwin_run import load
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction

def main():
    report=ROOT/'reports/real-transfer-v20';checks=0;sanity=[]
    def require(value,message):
        nonlocal checks
        if not value:raise AssertionError(message)
        checks+=1
    for case in CASES:
        data=phystwin(case,240);prefix,_=load(case,240,False)
        model=ExternalFEM(prefix,'cpu',field=False,nonconvex=True,materials=6)
        row=dict(case=case,methods={});start=data['train_end']
        for method in ('neural','analytic'):
            folder=ROOT/'runs/real-transfer-v19/fits'/LABEL/case/method
            meta=json.loads((folder/'complete.json').read_text())
            require(sha(folder/'prediction.npz')==meta['prediction_sha256'],'Modified prediction')
            require(sha(folder/'fit.json')==meta['fit_sha256'],'Modified fit')
            require(meta['fit_end']==start,'Incorrect training boundary')
            with np.load(folder/'prediction.npz') as f:p=f['predicted']
            result=tetra_sanity(data['nodes'],p,model.elements.numpy())
            require(result['finite_frames']==result['total_frames'],'Incomplete rollout')
            require(result['minimum_volume_ratio']>.029,'Singular/inverted FEM')
            dense=expand_prediction(data,p)[:, :data['evaluation']['surface_count']]
            below=np.maximum(-dense[start:,:,2],0)*1000
            result.update(plane_max_penetration_mm=float(below.max()),plane_fraction_below_2mm=float((below>2).mean()))
            row['methods'][method]=result
        native=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
        with native.open('rb') as f:p=np.asarray(pickle.load(f))@data['evaluation']['transform']
        below=np.maximum(-p[start:,:data['evaluation']['surface_count'],2],0)*1000
        row['methods']['official_phystwin']=dict(plane_max_penetration_mm=float(below.max()),plane_fraction_below_2mm=float((below>2).mean()),
            note='No native tetrahedral inversion score: method has a different physical representation.')
        sanity.append(row)
    for source,target in PAIRS:
        for method in ('neural','analytic'):
            source_dir=ROOT/'runs/real-transfer-v19/fits'/LABEL/source/method
            folder=BASE/'frozen-transfer/refined-frozen'/f'{source}--{target}'/method
            meta=json.loads((folder/'complete.json').read_text())
            require(meta['source_fit_sha256']==sha(source_dir/'fit.json'),'Wrong source fit')
            require(meta['target_post_initial_geometry_used'] is False,'Target trajectory used')
            require(meta['prediction_sha256']==sha(folder/'prediction.npz'),'Modified transfer')
            with np.load(source_dir/'best.npz') as f:old=f['theta']
            with np.load(folder/'prediction.npz') as f:new=f['theta']
            # Source has one hand; target one or two. Tail contains floor,
            # material field, fixed field slot and neural constitutive parameters.
            target_hands=2 if target.startswith('double') else 1
            require(np.array_equal(old[2:],new[1+target_hands:]),'Transferred material changed')
            require(new[0]==old[0],'Gravity/material ratio changed')
        native_dir=BASE/'native-transfer'/f'{source}--{target}'
        meta=json.loads((native_dir/'complete.json').read_text())
        require(meta['target_post_initial_geometry_used'] is False,'Native target fitting')
        with (native_dir/'input.pkl').open('rb') as f:bundle=pickle.load(f)
        require(not bundle['object_visibilities'].any() and not bundle['object_motions_valid'].any(),'Native target observations enabled')
        require(np.all(bundle['object_points']==bundle['object_points'][0]),'Native future geometry leaked')
    for source in ('single_lift_sloth','single_lift_zebra'):
        meta=json.loads((BASE/'native-transfer'/f'{source}--{source}'/'complete.json').read_text())
        require(meta['replay_max_abs_m']<.0005,'Native self-replay mismatch exceeds 0.5 mm')
    summary=json.loads((report/'summary.json').read_text())
    scores=json.loads((report/'refined-scores.json').read_text())
    for method in ('neural','analytic','official_phystwin','position_ridge'):
        for metric in ('tracking_mm','displacement_mm','pair_distance_change_mm'):
            value=sum(row['results'][method]['manual'][metric] for row in scores['rows'])/len(scores['rows'])
            require(abs(value-summary['standard_prediction'][method][metric])<1e-9,'Aggregate arithmetic mismatch')
    write(report/'final-audit.json',dict(passed=True,checks=checks,sanity=sanity,
        scope='Independent aggregate arithmetic and frozen-input/parameter checks, not an independent implementation of every optimizer. Plane checks use dataset z=0; no measured-force accuracy established.'))
    print('PASSED',checks,'checks')

if __name__=='__main__':main()
