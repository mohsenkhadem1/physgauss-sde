"""Read-only aggregation of sealed bowl predictions and retained outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('reports/gap-first-v01/bowl')

def clustered_mean_interval(values,worlds,seed=1107901):
    unique=np.unique(worlds)
    means=np.array([np.mean(np.asarray(values)[np.asarray(worlds)==w]) for w in unique])
    rng=np.random.default_rng(seed)
    sampled=means[rng.integers(len(means),size=(5000,len(means)))].mean(1)
    return dict(mean=float(means.mean()),cluster_bootstrap95=np.quantile(sampled,[.025,.975]).tolist(),independent_worlds=len(unique))

def main():
    path=ROOT/'case-results.json';rows=json.loads(path.read_text())
    prospective=ROOT/'prospective-predictions.json';seal=json.loads((ROOT/'prediction-seal.json').read_text())
    assert hashlib.sha256(prospective.read_bytes()).hexdigest()==seal['sha256']
    summary=dict(scope='Finite nonlinear contact posterior conditional on known geometry, density, damping, prescribed tool motion, and contact law; no RGB reconstruction or native-framework accuracy claim',
        rows=len(rows),worlds_per_cohort=16,independent_parameter_worlds_total=16,paired_meshes_same_worlds=True,noise_replicates_per_world=2,
        physical_models='Neo-Hookean soft solid in open hemispherical rigid bowl; surface penalty contact and regularized friction',
        groups={},paired={},source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        caveats=['Matched control reuses the same finite forward dictionary by design, not independent real-world validation',
                 'Refined-mesh cohort is a model-discretization stress test, not certified continuum ground truth',
                 'Intervals describe atomic finite-hypothesis uncertainty; unknown model discrepancy is not automatically calibrated',
                 'All action choices and risks were sealed before held-out target scoring',
                 'Bootstrap resamples16 material/contact worlds, preserving paired noise replicates; descriptive finite-cohort uncertainty'])
    metrics=['map_rms_mm','posterior_mean_rms_mm','predicted_failure_probability','actual_failure','brier','coordinate_coverage90','coordinate_width90_mm']
    for body in ('bowl','tool'):
        metrics.extend([f'{body}_reaction_mean_vector_rms_N',f'{body}_reaction_map_vector_rms_N',f'{body}_reaction_coordinate_coverage90',f'{body}_reaction_coordinate_width90_N'])
    summary['caveats'].append('Secondary resolution mixture includes the finer reference by construction, prior0.5 per resolution; it is a controlled comparator, not unknown-model robustness')
    for model in ('fem','resolution-mixture'):
        stages=['initial','fixed_push','fixed_tilt','active']+(['shared_coarse_active'] if model=='resolution-mixture' else [])
        for cohort in ('matched','mesh-mismatch'):
            for stage in stages:
                group=[r for r in rows if r['model']==model and r['cohort']==cohort and r['stage']==stage]
                worlds=[r['hypothesis'] for r in group]
                out={key:clustered_mean_interval([r[key] for r in group],worlds) for key in metrics}
                accepted=[r for r in group if r['predicted_failure_probability']<=.2]
                out['reporting_threshold_0_2']=dict(accepted=len(accepted),total=len(group),
                    actual_failure_rate=(float(np.mean([r['actual_failure'] for r in accepted])) if accepted else None),
                    mean_predicted_failure_probability=(float(np.mean([r['predicted_failure_probability'] for r in accepted])) if accepted else None))
                actual=[]
                for r in group:
                    if stage=='initial':actual.append('none')
                    elif stage=='fixed_push':actual.append('calibration_push')
                    elif stage=='fixed_tilt':actual.append('calibration_tilt')
                    elif stage=='shared_coarse_active':
                        primary=next(z for z in rows if z['model']=='fem' and z['cohort']==cohort and z['hypothesis']==r['hypothesis'] and z['replicate']==r['replicate'] and z['stage']=='active')
                        actual.append(primary['chosen_action'])
                    else:actual.append(r['chosen_action'])
                out['observed_action_counts']={name:actual.count(name) for name in ('none','calibration_push','calibration_tilt')}
                summary['groups'][model+'/'+cohort+'/'+stage]=out
            by={(r['hypothesis'],r['replicate'],r['stage']):r for r in rows if r['model']==model and r['cohort']==cohort}
            for baseline in ('initial','fixed_push','fixed_tilt'):
                out={}
                for metric in metrics:
                    differences=[];worlds=[]
                    for world in range(16):
                        for rep in range(2):
                            differences.append(float(by[world,rep,'active'][metric])-float(by[world,rep,baseline][metric]));worlds.append(world)
                    out[metric]=clustered_mean_interval(differences,worlds)
                summary['paired'][model+'/'+cohort+'/active-minus-'+baseline]=out
    for cohort in ('matched','mesh-mismatch'):
        by={(r['model'],r['hypothesis'],r['replicate'],r['stage']):r for r in rows if r['cohort']==cohort}
        out={}
        for metric in metrics:
            differences=[];worlds=[]
            for world in range(16):
                for rep in range(2):
                    differences.append(float(by['resolution-mixture',world,rep,'shared_coarse_active'][metric])-float(by['fem',world,rep,'active'][metric]));worlds.append(world)
            out[metric]=clustered_mean_interval(differences,worlds)
        summary['paired'][cohort+'/mixture-minus-coarse-same-active-evidence']=out
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    for key,value in summary['groups'].items():
        print(key,'MAP RMS mm',round(value['map_rms_mm']['mean'],3),'fail',round(value['actual_failure']['mean'],3),'Brier',round(value['brier']['mean'],3))

if __name__=='__main__':main()
