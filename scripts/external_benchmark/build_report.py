"""Build readable report and scientific plots from the complete frozen study."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.external_benchmark.common import *

def main():
    data=json.loads((REPORT/'results.json').read_text());summary=data['summaries'];rows=data['sequences']
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
                text.append(f'{method:30} {fmt("coordinate_rmse_mm"):>14} {fmt("aligned_shape_rmse_mm"):>12} {fmt("manual_track_mean_mm"):>13} {result["failed_frames"]:>14} {result["sequences_present"]:>4}/{result["sequences_expected"]:<5}')
    text+=['','PhysTwin per-sequence manual marker errors (mm)',f'{"Case":28} {"Our FEM":>12} {"Official PT":>12} {"Ridge":>12}']
    for row in rows:
        if row['dataset']!='PhysTwin':continue
        values=[]
        for method in ['material_world_fem','official_phystwin','position_ridge']:
            value=row['results'].get(method,{}).get('summary',{}).get('manual_track_mean_mm');values.append('-' if value is None else f'{value:.3f}')
        text.append(f'{row["name"]:28} {values[0]:>12} {values[1]:>12} {values[2]:>12}')
    jobs=data['job_status'];repairs=[j for j in jobs if j.get('repair_attempted')]
    text+=['','Execution coverage',f'{sum(j["state"]=="complete" for j in jobs)}/{len(jobs)} planned fitting jobs completed.',
        f'{len(repairs)} jobs used the declared prefix-only feasible-start repair; {sum(j["state"]=="complete" for j in repairs)} of those completed.',
        'Original failed initialization logs remain in the job records. A job completing',
        'means it saved a fitted checkpoint and prediction; it does not prove global',
        'optimizer convergence or correct material recovery. Per-fit optimizer stopping',
        'reasons are retained in results.json.']
    for dataset in ['PhysTwin','SoftVTBench']:
        subset=[r for r in rows if r['dataset']==dataset]
        fractions=[r['nonrigid_to_total_motion_ratio'] for r in subset]
        outside=[r['control_outside_prefix_box_fraction'] for r in subset]
        text+=[f'{dataset}: median nonrigid/total motion ratio {np.median(fractions):.4f}; mean fraction of forecast frames with tool coordinates outside prefix ranges (+1 mm margin) {np.mean(outside):.3f}.']
    text+=['','Adaptive release integration',
    'The adaptive_ FEM rows are post-freeze numerical refinements of every selected',
    'SoftVTBench episode, with identical fitted parameters. Substeps increase only',
    'when the nonlinear solver fails. No target error selects a step size. Original',
    'fixed-step results remain in the table above. See NUMERICAL_AMENDMENT.txt.',
    'A separate development time-step study did NOT establish converged release',
    'dynamics: the original fine-step solver failed; a positive-volume diagnostic',
    'extension ran but showed up to 17.5 mm change against a finer discretization.',
    'Adaptive finite predictions therefore demonstrate stability, not temporal',
    'convergence or a validated dynamics reference.',
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
    'Nine numerical tests cover analytic energy derivatives, Hessians, material',
    'features, implicit adjoints, contact and inertia. The in-memory target-poisoning',
    'audit confirms that future truth/manual markers cannot change predictor inputs.',
    'frozen-protocol.json lists exact jobs and core source hashes; results.json',
    'contains every frame score, job status, fitted parameters and prediction hashes.',
    'Raw datasets and large checkpoints remain local under runs/. Videos are',
    'shared-scale geometry playbacks with original frame rates and fitting/test',
    'labels. They are not raw RGB or photorealistic renderings.','',
    'References: https://github.com/TuojingAI/SoftVTBench',
    '            https://huggingface.co/datasets/Arthur12137/SoftVTBench',
    '            https://github.com/Jianghanxiao/PhysTwin',
    '            https://huggingface.co/datasets/Jianghanxiao/PhysTwin']
    (REPORT/'RESULTS.txt').write_text('\n'.join(text)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    specs=[('PhysTwin','full','manual_track_mean_mm','Manual marker error (mm)'),('SoftVTBench','held_grasp','coordinate_rmse_mm','Commanded-hold position RMSE (mm)'),('SoftVTBench','full','coordinate_rmse_mm','Full-suffix position RMSE (mm)')]
    methods=['control_rigid','position_ridge','material_world_fem','no_world_potential_fem']
    labels={'control_rigid':'Rigid','position_ridge':'Ridge','material_world_fem':'Material + world FEM','no_world_potential_fem':'No extra world potential','official_phystwin':'Official PhysTwin','adapted_phystwin':'Adapted PhysTwin','adaptive_material_world_fem':'FEM (adaptive steps)','adaptive_no_world_potential_fem':'Ablation (adaptive steps)'}
    for ax,(dataset,segment,metric,title) in zip(axes,specs):
        chosen=methods if dataset=='PhysTwin' else ['control_rigid','position_ridge','adaptive_material_world_fem','adaptive_no_world_potential_fem']
        allmethods=chosen+(['official_phystwin'] if dataset=='PhysTwin' else ['adapted_phystwin'])
        present=[m for m in allmethods if metric in summary[dataset][segment][m]['mean']]
        values=[summary[dataset][segment][m]['mean'][metric] for m in present]
        colors=['#176b87' if m in ['material_world_fem','adaptive_material_world_fem'] else '#c6ced3' for m in present]
        ax.barh([labels[m] for m in present],values,color=colors)
        ax.invert_yaxis();ax.set_title(dataset);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(values):ax.text(value,i,f' {value:.1f}',va='center',fontsize=9)
        ax.set_xlim(0,max(values)*1.18)
    fig.suptitle('Reserved external sequences — per-sequence means, lower is better')
    fig.savefig(REPORT/'comparison.png',dpi=180);plt.close(fig)
    (REPORT/'STATUS.txt').write_text('Primary study complete. See RESULTS.txt, results.json and videos. Raw datasets remain local. Publication verification/push may still be pending.\n')
    print(REPORT/'RESULTS.txt')

if __name__=='__main__':main()
