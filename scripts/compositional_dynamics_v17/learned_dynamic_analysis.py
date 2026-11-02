"""All-pair descriptive transfer errors; no qualified-truth or inference claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics
from scripts.compositional_dynamics_v17.learned_dynamic_pilot import PLAN,RAW,RESULT,REPORT,ROOT,verify,sha,write


def main():
    plan=json.loads(PLAN.read_text());verify(plan)
    results=json.loads(RESULT.read_text())
    if not results.get('terminal'):raise ValueError('Wait for terminal accounting, including missing cases')
    records={row['task']['id']:row for row in results['rows']};pairs=[]
    hashes={str(PLAN.relative_to(ROOT)):sha(PLAN),str(RESULT.relative_to(ROOT)):sha(RESULT)}
    for family in ('mooney_rivlin','yeoh'):
        for case in ('release','drop'):
            identifiers=[f'{family}-{case}-{method}' for method in ('teacher','frozen_shared')]
            rows=[records.get(key) for key in identifiers]
            pair=dict(family=family,case=case,ids=identifiers,
                      complete=all(row is not None and row['completed'] for row in rows),
                      expected_duration_s=plan['settings']['duration'][case],
                      methods={method:({key:row.get(key) for key in ('completed','error','failure_time_s','saved_frames','minimum_J','maximum_relative_energy_drift','maximum_dense_penetration_m')} if row else dict(completed=False,error='Missing declared run')) for method,row in zip(('teacher','frozen_shared'),rows)})
            files=[RAW/key/'trajectory.npz' for key in identifiers]
            if all(path.exists() for path in files):
                hashes.update({str(path.relative_to(ROOT)):sha(path) for path in files})
                with np.load(files[0]) as truth,np.load(files[1]) as prediction:
                    if not np.array_equal(truth['rest_nodes'],prediction['rest_nodes']) or not np.array_equal(truth['mass_row_sums'],prediction['mass_row_sums']):raise ValueError('Matched known material geometry/mass required')
                    count=min(len(truth['times_s']),len(prediction['times_s']))
                    if count>=2:
                        # A failed teacher is only a common-prefix diagnostic, never a complete reference.
                        times=truth['times_s'] if rows[0]['completed'] else truth['times_s'][:count]
                        pred_count=min(count,len(times))
                        metrics=matched_sequence_metrics(truth['rest_nodes'],times,truth['rest_nodes'][None]+truth['u'][:len(times)],prediction['times_s'][:pred_count],prediction['rest_nodes'][None]+prediction['u'][:pred_count],truth['mass_row_sums'],characteristic_length_m=.06)
                        metrics['sampled_comparison_prefix_complete']=metrics.pop('complete')
                        metrics['complete']=pair['complete']
                        metrics['reference_available_frames']=len(truth['times_s'])
                        metrics['expected_frames']=round(pair['expected_duration_s']/plan['settings']['sample_interval_s'])+1
                        metrics['frame_coverage']=pred_count/metrics['expected_frames']
                        metrics['first_missing_time_s']=None if pair['complete'] else pred_count*plan['settings']['sample_interval_s']
                        metrics['aggregation_scope']='Entire declared trajectory' if pair['complete'] else 'Common available prefix only; failed full trajectory retained'
                        metrics['physical_reference_qualified']=False
                        metrics['teacher_trajectory_complete']=bool(rows[0]['completed'])
                        metrics['common_saved_duration_s']=float(prediction['times_s'][pred_count-1]-prediction['times_s'][0])
                        metrics['declared_duration_coverage']=metrics['common_saved_duration_s']/pair['expected_duration_s']
                        pair['metrics']=metrics
            pairs.append(pair)
    out=dict(scope='Known-latent frozen bulk-law transfer on development dynamics. Analytic-law trajectory shares an unqualified discretization; it is not independent ground truth. No physical inference or native-framework superiority claim.',expected_pairs=4,completed_pairs=sum(pair['complete'] for pair in pairs),pairs=pairs,input_sha256=hashes,
             source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py']})
    write(REPORT/'learned-dynamic-pilot-analysis.json',out)
    print(json.dumps(dict(completed_pairs=out['completed_pairs'],pairs=[dict(family=p['family'],case=p['case'],complete=p['complete'],world_rms_mm=(p['metrics']['world_point_rms_m']*1000 if 'metrics' in p else None)) for p in pairs])))


if __name__=='__main__':main()
