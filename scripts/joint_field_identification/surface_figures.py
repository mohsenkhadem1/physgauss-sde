"""Final complete-cohort surface plots; no paper source edits or case selection."""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
from . import surface as bank
ROOT=bank.ROOT;REPORT=ROOT/'reports/joint-field-surface-v04';OUT=ROOT/'output/joint-field-surface-v04'
POLICIES=['active','fixed','fixed-view','fixed-probe','random'];POP=['matched','continuous-coarse','offgrid-fine','force-prior']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    deadline=datetime.datetime.fromisoformat(json.loads((REPORT/'protocol.json').read_text())['deadline_utc'])
    while True:
        p=REPORT/'summary.json';doc=json.loads(p.read_text()) if p.exists() else {}
        if doc.get('complete'):break
        if not args.wait:raise RuntimeError('Only completed final cohorts may generate these figures')
        if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Incomplete cohort at deadline')
        time.sleep(30)
    summary=doc['summary'];OUT.mkdir(parents=True,exist_ok=True)
    # Static scientific plots with exact saved values, no illustrative improvement.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in (Path.home()/'.local/share/fonts/arial').glob('*.TTF'):font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42})
    colours=dict(active='#007d8a',fixed='#b45b21',**{'fixed-view':'#5575a3','fixed-probe':'#9864a5','random':'#777777'})
    fig,axes=plt.subplots(2,2,figsize=(10,6.2),layout='constrained')
    for ax,(key,label) in zip(axes.flat,[('hidden_rmse_mm','Unobserved motion error (mm)'),('changed_load_rmse_mm','Changed-load error (mm)'),('material_relative_rmse','Material field relative error'),('force_rmse_N','Unknown load error (N)')]):
        for policy in POLICIES:
            rr=[r for r in summary if r['population']=='matched' and r['policy']==policy]
            ax.plot([r['step'] for r in rr],[r[key] for r in rr],'-o',color=colours[policy],label=policy.replace('-',' '),lw=2 if policy=='active' else 1.4,ms=4)
        ax.set_xlabel('Acquired scalar measurements');ax.set_ylabel(label);ax.set_xticks(range(5));ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
    fig.savefig(OUT/'acquisition.png',dpi=220);fig.savefig(OUT/'acquisition.pdf');plt.close(fig)
    # Compact main-paper prediction curves; full four-panel diagnostic retained.
    fig,axes=plt.subplots(1,2,figsize=(10,3.1),layout='constrained')
    for ax,(key,label) in zip(axes,[('hidden_rmse_mm','Unobserved motion error (mm)'),('changed_load_rmse_mm','Changed-load error (mm)')]):
        for policy in POLICIES:
            rr=sorted([r for r in summary if r['population']=='matched' and r['policy']==policy],key=lambda r:r['step'])
            assert len(rr)==5 and all(r['n']==128 for r in rr)
            ax.plot([r['step'] for r in rr],[r[key] for r in rr],'-o',color=colours[policy],label=policy.replace('-',' '),lw=2 if policy=='active' else 1.4,ms=4)
        ax.set_xlabel('Acquired scalar measurements');ax.set_ylabel(label);ax.set_xticks(range(5));ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
    fig.savefig(OUT/'acquisition-prediction.png',dpi=240);fig.savefig(OUT/'acquisition-prediction.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
    for ax,(key,label) in zip(axes,[('coverage90','Prediction coverage'),('material_coverage90','Material coverage'),('force_coverage90','Load coverage')]):
      for j,pop in enumerate(POP):
        rr=[r for r in summary if r['population']==pop and r['policy']=='active' and r['step'] in (0,4)]
        if len(rr)==2:ax.plot([j-.12,j+.12],[100*r[key] for r in rr],'-o',color=['#007d8a','#5575a3','#b45b21','#9864a5'][j])
      ax.axhline(90,color='black',ls='--',lw=.8);ax.set_ylim(0,102);ax.set_xticks(range(4),['Matched','Off-grid','Fine mesh','Force prior'],rotation=20);ax.set_ylabel(label+' (%)');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'coverage.png',dpi=220);fig.savefig(OUT/'coverage.pdf');plt.close(fig)
    # Single-column main-paper panel; the three-panel figure stays available.
    with plt.rc_context({'font.family':'Arial','font.size':10.5}):
        fig,ax=plt.subplots(figsize=(3.6,2.8),layout='constrained')
        for j,pop in enumerate(POP):
            before=next(r for r in summary if r['population']==pop and r['policy']=='active' and r['step']==0)
            after=next(r for r in summary if r['population']==pop and r['policy']=='active' and r['step']==4)
            values=[100*before['coverage90'],100*after['coverage90']]
            ax.plot([j-.12,j+.12],values,color='#777777',lw=1.3,zorder=2)
            ax.plot(j-.12,values[0],'o',mfc='white',mec='black',ms=5,label='Before' if j==0 else None,zorder=3)
            ax.plot(j+.12,values[1],'o',mfc='#007d8a',mec='black',ms=5,label='After 4 measurements' if j==0 else None,zorder=3)
        ax.axhline(90,color='black',ls='--',lw=.9)
        ax.text(3.3,94,'Nominal 90%',ha='right',va='bottom',fontsize=9)
        ax.set_ylim(70,100);ax.set_yticks([70,80,90,100]);ax.set_xlim(-.5,3.5)
        ax.set_xticks(range(4),['Matched','Off-grid','Fine mesh','Force\nprior'])
        ax.set_ylabel('Prediction coverage (%)')
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9,handletextpad=.3,columnspacing=.7)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/'coverage-prediction.png',dpi=240);fig.savefig(OUT/'coverage-prediction.pdf');plt.close(fig)
    # Generated LaTeX tables retain all policy outcomes.
    text=['\\begin{tabular}{lrrrr}\\toprule','Policy & Hidden & New load & Material & Force\\\\',' & mm & mm & rel. & N\\\\\\midrule']
    initial=[r for r in summary if r['population']=='matched' and r['policy']=='active' and r['step']==0]
    chosen=initial+[r for r in summary if r['population']=='matched' and r['policy'] in POLICIES and r['step']==4]
    for r in chosen:
        name='Initial' if r['step']==0 else r['policy'].replace('-',' ').title()
        text.append(f"{name} & {r['hidden_rmse_mm']:.3f} & {r['changed_load_rmse_mm']:.3f} & {r['material_relative_rmse']:.3f} & {r['force_rmse_N']:.3f}\\\\")
    text+=['\\bottomrule\\end{tabular}'];(REPORT/'acquisition-table.tex').write_text('\n'.join(text))
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPORT/'summary.json',REPORT/'all-case-metrics.csv',REPORT/'protocol.json',Path(__file__)]}
    manifest=dict(complete=True,source_sha256=sources,counts=doc['counts'],selection='Entire completed cohort; matched acquisition curves, all4populations coverage; no case filtering',text='Arial, black; policy curves use colors',coverage_prediction_axis='Explicit70–100percent zoom; ticks70/80/90/100. Point values unchanged; fullcoverage panel remains0–102percent.',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT/'aggregate-figure-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
