"""All-case comparison; no example selection or cross-dataset metric pooling."""
import gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.contact_benchmark_v12.common import REPORT
from scripts.contact_benchmark_v12.model_selection_report import METRIC,SELECTED,complete

def main():
    with gzip.open(REPORT/'model-selection-results.json.gz','rt') as f:data=json.load(f)
    assert data['accounting_complete'],'Plot requires all declared cases'
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    colors={'development':'#6b7e91','regression':'#24796b','untouched_confirmation':'#c08a36','fresh_confirmation':'#735a9a'}
    labels={'development':'Development','regression':'Regression','untouched_confirmation':'Earlier cohort','fresh_confirmation':'Later cohort'}
    for ax,(dataset,metric) in zip(axes,METRIC.items()):
        prior='feasible_contact_scene_state' if dataset=='SoftVTBench' else 'feasible_object_frame';all_values=[];invalid=0
        for cohort,rows in data['sequences'].items():
            pairs=[]
            for row in rows:
                if row['dataset']!=dataset:continue
                a=row['results'][prior]['summary'];b=row['results'][SELECTED]['summary']
                if not complete(a,metric) or not complete(b,metric):invalid+=1;continue
                pairs.append((a[metric],b[metric],row['name']))
            if pairs:
                ax.scatter([p[0] for p in pairs],[p[1] for p in pairs],s=30 if dataset=='SoftVTBench' else 50,c=colors[cohort],label=labels[cohort],alpha=.8,edgecolors='white',linewidths=.4,zorder=3)
                all_values.extend(v for p in pairs for v in p[:2])
        limit=max(all_values,default=1)*1.2;ax.plot([0,limit],[0,limit],color='#999999',ls='--',lw=1)
        ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        n=data['overall_datasets'][dataset]['expected'];ax.set_title(f'{dataset}: {n-invalid}/{n} complete pairs',fontsize=11)
        ax.set_xlabel('Previous guarded forecast error (mm)');ax.set_ylabel('Selected-model forecast error (mm)')
        ax.text(.03,.96,'Below diagonal = improvement',transform=ax.transAxes,va='top',fontsize=9)
        if invalid:ax.text(.03,.89,f'{invalid} incomplete pairs remain in the report',transform=ax.transAxes,va='top',fontsize=8,color='#a03a31')
    axes[0].legend(fontsize=8,loc='lower right',frameon=True)
    fig.suptitle('All-case model selection: post-audit exploratory evidence',fontsize=14)
    fig.text(.5,.025,'Development cases included. SoftVT uses coordinate RMSE; PhysTwin uses manual-marker mean error. Metrics are not pooled.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.94]);fig.savefig(REPORT/'model-selection-comparison.png',dpi=180);fig.savefig(REPORT/'model-selection-comparison.svg');plt.close(fig)

if __name__=='__main__':main()
