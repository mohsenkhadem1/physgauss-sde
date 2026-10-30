"""Build readable report and scientific plots from the complete frozen study."""
import sys,json,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.external_benchmark.common import *

def main():
    data=json.loads((REPORT/'results.json').read_text() if (REPORT/'results.json').exists() else gzip.decompress((REPORT/'results.json.gz').read_bytes()));summary=data['summaries'];rows=data['sequences']
    if data['partial']:raise ValueError('Cannot label partial results final')
    text=['EXTERNAL DEFORMATION BENCHMARK v11','Completed: '+datetime.now(timezone.utc).isoformat(),'','What was done',
    'The material-reference/world-boundary model now runs on two external datasets.',
    'It learns physical ratios from an observed motion prefix and predicts the later',
    'object shape using recorded future tool positions, without true force or material',
    'labels. Initial geometry is provided. This is future geometry prediction after',
    'per-sequence fitting, not zero-shot prediction or an RGB reconstruction result.','',
    'Cohort: 40 SoftVTBench episodes (2 suites x 2 reserved tasks x 10 episodes),',
    'and 3 reserved PhysTwin sequences. Development cases are excluded from these',
    'results. Configurations were frozen and pushed before primary scoring.','',
    'Methods: persistence, rigid tool transport, prefix-calibrated rigid transport,',
    'position ridge regression, full FEM,',
    'FEM without its additional world-space Gaussian potential, official released',
    'PhysTwin predictions on its data, and adapted native PhysTwin on SoftVTBench.','',
    'Mean sequence errors (mm; lower is better)']
    for dataset,parts in summary.items():
        for part,methods in parts.items():
            text+=['',f'{dataset} / {"commanded_hold" if part=="held_grasp" else "commanded_release" if part=="released" else part}',f'{"Method":30} {"position RMSE":>14} {"shape RMSE":>12} {"manual track":>13} {"failed frames":>14} {"present":>10}']
            for method,result in methods.items():
                m=result['mean'];fmt=lambda key:f'{m[key]:.3f}' if key in m else '-'
                display={'physical_release_material_world_fem':'FEM: corrected release','physical_release_no_world_potential_fem':'Ablation: corrected release'}.get(method,method)
                text.append(f'{display:30} {fmt("coordinate_rmse_mm"):>14} {fmt("aligned_shape_rmse_mm"):>12} {fmt("manual_track_mean_mm"):>13} {result["failed_frames"]:>14} {result["sequences_present"]:>4}/{result["sequences_expected"]:<5}')
    findings=['Main findings']
    pt=summary['PhysTwin']['full']
    ours=pt['material_world_fem']['mean'].get('manual_track_mean_mm');ablated=pt['no_world_potential_fem']['mean'].get('manual_track_mean_mm');official=pt['official_phystwin']['mean'].get('manual_track_mean_mm')
    if all(v is not None for v in [ours,ablated,official]):
        findings += [f'PhysTwin: our mean manual-marker error is {ours:.2f} mm, versus {official:.2f} mm for the official reference.',f'The additional world-space potential improves our own ablation from {ablated:.2f} to {ours:.2f} mm ({100*(ablated-ours)/ablated:.1f}%) across these three cases.']
    st=summary['SoftVTBench']
    if 'physical_release_material_world_fem' in st['full']:
        a=st['full']['physical_release_material_world_fem'];hold=st['held_grasp']['physical_release_material_world_fem']['mean']
        findings += [f'SoftVTBench full-suffix position RMSE: corrected FEM {a["mean"]["coordinate_rmse_mm"]:.2f} mm; adapted PhysTwin {st["full"]["adapted_phystwin"]["mean"]["coordinate_rmse_mm"]:.2f} mm; calibrated rigid {st["full"]["prefix_calibrated_rigid"]["mean"]["coordinate_rmse_mm"]:.2f} mm.',f'Commanded-hold position RMSE: corrected FEM {hold["coordinate_rmse_mm"]:.2f} mm; adapted PhysTwin {st["held_grasp"]["adapted_phystwin"]["mean"]["coordinate_rmse_mm"]:.2f} mm; calibrated rigid {st["held_grasp"]["prefix_calibrated_rigid"]["mean"]["coordinate_rmse_mm"]:.2f} mm.',f'Corrected FEM produced {a["sequences_complete"]}/{a["sequences_expected"]} complete sequences with {a["failed_frames"]} failed forecast frames. This establishes finite forecasts, not converged release dynamics.']
    text[3:3]=findings+['']
    text+=['','PhysTwin per-sequence manual marker errors (mm)',f'{"Case":28} {"Our FEM":>12} {"Official PT":>12} {"Ridge":>12}']
    for row in rows:
        if row['dataset']!='PhysTwin':continue
        values=[]
        for method in ['material_world_fem','official_phystwin','position_ridge']:
            value=row['results'].get(method,{}).get('summary',{}).get('manual_track_mean_mm');values.append('-' if value is None else f'{value:.3f}')
        text.append(f'{row["name"]:28} {values[0]:>12} {values[1]:>12} {values[2]:>12}')
    text+=['','SoftVTBench per-group full-suffix position RMSE (mm)',f'{"Group":30} {"Corrected FEM":>14} {"Adapted PT":>12} {"Calibrated rigid":>18}']
    groups=sorted(set(r['name'].split('-demo')[0] for r in rows if r['dataset']=='SoftVTBench'))
    for group in groups:
        selected=[r for r in rows if r['name'].startswith(group+'-demo')];values=[]
        for method in ['physical_release_material_world_fem','adapted_phystwin','prefix_calibrated_rigid']:
            numbers=[r['results'].get(method,{}).get('summary',{}).get('coordinate_rmse_mm') for r in selected];numbers=[v for v in numbers if v is not None]
            values.append('-' if not numbers else f'{np.mean(numbers):.3f}')
        text.append(f'{group:30} {values[0]:>14} {values[1]:>12} {values[2]:>18}')
    jobs=data['job_status'];repairs=[j for j in jobs if j.get('repair_attempted')]
    text+=['','Execution coverage',f'{sum(j["state"]=="complete" for j in jobs)}/{len(jobs)} planned fitting jobs completed.',
        f'{len(repairs)} jobs used the declared prefix-only feasible-start repair; {sum(j["state"]=="complete" for j in repairs)} of those completed.',
        'Original failed initialization logs remain in the job records. A job completing',
        'means it saved a fitted checkpoint and prediction; it does not prove global',
        'optimizer convergence or correct material recovery. Per-fit optimizer stopping',
        'reasons are retained in results.json.gz (lossless JSON archive).']
    for dataset in ['PhysTwin','SoftVTBench']:
        subset=[r for r in rows if r['dataset']==dataset]
        fractions=[r['nonrigid_to_total_motion_ratio'] for r in subset]
        outside=[r['control_outside_prefix_box_fraction'] for r in subset]
        text+=[f'{dataset}: median nonrigid/total motion ratio {np.median(fractions):.4f}; mean fraction of forecast frames with tool coordinates outside prefix ranges (+1 mm margin) {np.mean(outside):.3f}.']
    text+=['','Release integration corrections',
    'The physical_release_ rows remove the artificial static rest-position gauge',
    'spring during inertial release. That spring is a numerical device, not a',
    'physical restoring force. Inertia already fixes the free rigid-body modes.',
    'Every fitted parameter and the physical energy terms stay unchanged. A',
    'translated-body free-fall test checks the correction. It is applied to all',
    '40 episodes and both FEM variants; earlier variants remain in the table.',
    'The adaptive_ FEM rows are post-freeze numerical refinements of every selected',
    'SoftVTBench episode, with identical fitted parameters. Substeps increase only',
    'when the nonlinear solver fails. No target error selects a step size. Original',
    'fixed-step results remain in the table above. See NUMERICAL_AMENDMENT.txt.',
    'The original release formulation did NOT establish converged release',
    'dynamics: the original fine-step solver failed; a positive-volume diagnostic',
    'extension ran but showed up to 17.5 mm change against a finer discretization.',
    'Adaptive finite predictions therefore demonstrate stability, not temporal',
    'convergence or a validated dynamics reference.',
    'The corrected-release development audit also failed to obtain a complete',
    'fine-step reference. Its translated-body free-fall check passes, but general',
    'release temporal convergence remains unestablished.',
    '','Interpretation and scope',
    'Position RMSE measures location plus deformation. Shape RMSE removes the best',
    'rigid alignment, isolating shape difference. PhysTwin manual marker distance',
    'and one-way L1 Chamfer are also reported; dense processed track errors are an',
    'additional diagnostic and can include tracking noise. Different metrics must',
    'not be substituted when a method loses on one but wins on another.',
    'SoftVTBench phase intervals use only tool commands and positions. The JSON key',
    'held_grasp means commanded hold, not verified physical attachment: an object',
    'can slip or drop while the gripper command remains closed. released likewise',
    'means commanded release. These phases must not be confused with true contact.',
    'Full-suffix results remain primary. No future target error was used to exclude',
    'frames. No new container/contact geometry is supplied after release.',
    'The FEM assumes quasistatic held-grasp motion and treats the initial observed',
    'shape as stress-free. PhysTwin uses spring-mass dynamics and its own mesh.',
    'These representation differences are material to interpretation.',
    'These runs are point-estimate predictions; they do not establish calibrated',
    'physical uncertainty, active probing superiority, or unseen-force-family OOD.',
    'SoftVTBench observations are privileged sparse node geometry, not noisy camera',
    'tracks. Its official manipulation-policy success-rate leaderboard is not used.',
    'All means weight sequences equally. Errors condition on finite predictions;',
    'missing methods and failed frames are separately retained. Forty episodes',
    'cover two pastry assets in four suite/task groups, not forty independent objects.','',
    'Reliability and reproduction',
    'Ten numerical tests cover analytic energy derivatives, Hessians, material',
    'features, implicit adjoints, contact and inertia. The in-memory target-poisoning',
    'audit confirms that future truth/manual markers cannot change predictor inputs.',
    'frozen-protocol.json lists exact jobs and core source hashes; results.json.gz',
    'contains every frame score, job status, fitted parameters and prediction hashes.',
    'Raw datasets and large checkpoints remain local under runs/. Videos are',
    'shared-scale geometry playbacks with original frame rates and fitting/test',
    'labels. They are not raw RGB or photorealistic renderings.',
    'Videos: primary-double_stretch_sloth.mp4; physical-release-object-soft-task1-demo0.mp4;',
    'physical-release-spatial-soft-task2-demo0.mp4. The additional',
    'adaptive-spatial-soft-task1-demo2.mp4 compares a selected numerical failure',
    'case before/after adaptive stepping; it is a failure diagnostic, not a randomly',
    'selected representative example.','',
    'References: https://github.com/TuojingAI/SoftVTBench',
    '            https://huggingface.co/datasets/Arthur12137/SoftVTBench',
    '            https://github.com/Jianghanxiao/PhysTwin',
    '            https://huggingface.co/datasets/Jianghanxiao/PhysTwin']
    (REPORT/'RESULTS.txt').write_text('\n'.join(text)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    specs=[('PhysTwin','full','manual_track_mean_mm','Manual marker error (mm)'),('SoftVTBench','held_grasp','coordinate_rmse_mm','Commanded-hold position RMSE (mm)'),('SoftVTBench','full','coordinate_rmse_mm','Full-suffix position RMSE (mm)')]
    methods=['prefix_calibrated_rigid','position_ridge','material_world_fem','no_world_potential_fem']
    labels={'control_rigid':'Rigid','prefix_calibrated_rigid':'Calibrated rigid','position_ridge':'Ridge','material_world_fem':'Material + world FEM','no_world_potential_fem':'No extra world potential','official_phystwin':'Official PhysTwin','adapted_phystwin':'Adapted PhysTwin','physical_release_material_world_fem':'FEM (corrected release)','physical_release_no_world_potential_fem':'Ablation (corrected release)'}
    for ax,(dataset,segment,metric,title) in zip(axes,specs):
        chosen=methods if dataset=='PhysTwin' else ['prefix_calibrated_rigid','position_ridge','physical_release_material_world_fem','physical_release_no_world_potential_fem']
        allmethods=chosen+(['official_phystwin'] if dataset=='PhysTwin' else ['adapted_phystwin'])
        present=[m for m in allmethods if metric in summary[dataset][segment][m]['mean']]
        values=[summary[dataset][segment][m]['mean'][metric] for m in present]
        colors=['#176b87' if m in ['material_world_fem','physical_release_material_world_fem'] else '#c6ced3' for m in present]
        ax.barh([labels[m] for m in present],values,color=colors)
        ax.invert_yaxis();ax.set_title(dataset);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(values):ax.text(value,i,f' {value:.1f}',va='center',fontsize=9)
        ax.set_xlim(0,max(values)*1.18)
    fig.suptitle('Reserved external sequences — per-sequence means, lower is better')
    fig.savefig(REPORT/'comparison.png',dpi=180);plt.close(fig)
    (REPORT/'STATUS.txt').write_text('Primary study complete. See RESULTS.txt, results.json and videos. Raw datasets remain local. Publication verification/push may still be pending.\n')
    print(REPORT/'RESULTS.txt')

if __name__=='__main__':main()
