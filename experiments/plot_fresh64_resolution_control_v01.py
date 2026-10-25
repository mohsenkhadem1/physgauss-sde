"""Plot audited scalar integration diagnostics from the three saved controls."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/fresh64-inner-resolution-control-v01'
REVIEW_SHA='717dd2b2ad8fae41f838be8d617a4f2448bad084d8afe194f3470009a6599400'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render(output):
    review_path=LOCAL/'independent-review.json'
    if digest(review_path)!=REVIEW_SHA:raise ValueError('Exact independent control review')
    review=json.loads(review_path.read_text())
    if review.get('passed')is not True or review.get('all_owned_PID_groups_absent')is not True:
        raise ValueError('Complete passed and closed control review')
    binding=review['worker_result'];result_path=ROOT/binding['path']
    if digest(result_path)!=binding['sha256']:raise ValueError('Exact reviewed scalar diagnostics')
    result=json.loads(result_path.read_text())
    labels=[('object05-nonlinear-normal-r0-inner','Nonlinear: inner refinement','inner'),
            ('object01-nonlinear-axial-r0-outer','Nonlinear: outer refinement','outer'),
            ('object01-linear-axial-r0-outer','Linear baseline: outer refinement','outer')]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True,layout='constrained')
    for ax,(key,title,axis)in zip(axes,labels):
        case=result['cases'][key]
        if case.get('passed')is not True:raise ValueError('All three unchanged controls required')
        rows=case['diagnostics']['successive'][axis]
        if [(r['coarse'],r['fine'])for r in rows]!=[(128,256),(256,512),(512,1024)]:
            raise ValueError('Exact original and added refinement sequence')
        y=[r['maximum_CDF_difference'] for r in rows]
        ax.plot(range(3),y,'o-',color='#197c78',linewidth=2)
        ax.axhline(.005,linestyle='--',color='#9b5b45',label='Unchanged limit: 0.005')
        ax.set_xticks(range(3),['128 → 256','256 → 512','512 → 1024'],rotation=20)
        ax.set_yscale('log');ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Largest change in posterior cumulative probability')
    axes[1].legend(loc='upper right',fontsize=8)
    fig.suptitle('Finer integration resolves three numerical failures',fontsize=14)
    fig.supxlabel('Same observations, physical model and accuracy limits; only integration resolution changes.\n'
        'Three numerical controls. These do not establish population calibration or physical prediction accuracy.',fontsize=9)
    target=output/'numerical-resolution-control.png';fig.savefig(target,dpi=180);plt.close(fig)
    proof=dict(independent_review_SHA256=REVIEW_SHA,result_SHA256=binding['sha256'],
        figure_SHA256=digest(target),raw_observations_or_physical_parameters_read=False,
        numerical_controls=3,no_population_calibration_or_prediction_claim=True)
    (output/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(render(p.parse_args().output)))
