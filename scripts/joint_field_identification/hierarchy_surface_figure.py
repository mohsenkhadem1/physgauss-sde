"""Measured surface-track hierarchy figures; independent acquisition branches.

Arial text stays black. Data colors identify conditions/models. No outcome-based
case selection, trajectory exaggeration, or changes to frozen numerical outputs.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from .hierarchy_surface import ROOT,REPORT,BASE,SCALES,SCHEDULES,fit_scale_mixture,update,verify_protocol
from . import surface as bank

OUT=ROOT/'output/joint-field-hierarchy-surface-v05'
CONDITION_COLORS=['#ADB7C1','#D5984F','#278C8A']
MODEL_ORDER=['fixed-nominal','variance-matched','hierarchical','oracle-scale']
MODEL_COLORS=['#9A9FA6','#D5984F','#278C8A','#6C7EA4']
MODEL_MARKERS=['s','D','o','^']


def style():
    font=Path.home()/'.local/share/fonts/arial/Arial.TTF'
    font_manager.fontManager.addfont(str(font))
    bold=font.with_name('Arial_Bold.TTF')
    if bold.exists():font_manager.fontManager.addfont(str(bold))
    plt.rcParams.update({'font.family':'Arial','font.size':9,'text.color':'black',
        'axes.labelcolor':'black','axes.edgecolor':'black','xtick.color':'black','ytick.color':'black',
        'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':180,
        'savefig.dpi':240,'hatch.linewidth':.5})


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),facecolor='white',bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/(name+'.png'),facecolor='white',bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#DDDDDD',linewidth=.55)
    ax.tick_params(length=3,width=.6,pad=2)


def collect():
    docs=[json.loads((REPORT/shape/f'case-{case:03d}.json').read_text()) for shape in ['block','step'] for case in range(64)]
    rng=np.random.default_rng(63001)
    boot=np.concatenate([rng.integers(0,64,(2048,64)),64+rng.integers(0,64,(2048,64))],axis=1)
    def values(model,condition,key):
        return np.array([next(r[key] for r in d['rows'] if r['model']==model and r['condition']==condition) for d in docs])
    def stats(model,condition,key,factor=1.):
        v=values(model,condition,key)*factor;lo,hi=np.quantile(v[boot].mean(1),[.025,.975])
        return dict(mean=float(v.mean()),ci95=[float(lo),float(hi)])
    return docs,values,stats


def bar_panel(ax,entries,colors,labels,title,ylabel,fmt):
    means=np.array([r['mean'] for r in entries]);lo=np.array([r['ci95'][0] for r in entries]);hi=np.array([r['ci95'][1] for r in entries])
    x=np.arange(len(entries));ax.bar(x,means,color=colors,width=.62,edgecolor='white',linewidth=.5,zorder=2)
    ax.errorbar(x,means,yerr=np.array([means-lo,hi-means]),fmt='none',ecolor='black',capsize=3,elinewidth=.9,capthick=.9,zorder=3)
    top=hi.max()*1.23;ax.set_ylim(0,top);ax.set_xlim(-.6,len(entries)-.4)
    for j,v in enumerate(means):ax.text(j,hi[j]+top*.025,fmt.format(v),ha='center',va='bottom',fontsize=9,color='black')
    ax.set_xticks(x,labels);ax.set_ylabel(ylabel,labelpad=3);ax.set_title(title,loc='left',fontweight='bold',pad=6);clean(ax)


def main():
    style();OUT.mkdir(parents=True,exist_ok=True);protocol,sha=verify_protocol();docs,values,stats=collect()
    assert all(d['protocol_sha256']==sha for d in docs)
    figuredata=dict(protocol_sha256=sha,cohort_cases=128,bootstrap='2048pairedcase resamples within geometry,seed63001',figures={})
    conditions=['initial','views','probes'];labels=['Initial','+4 views','+4 probes']
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.65));fig.subplots_adjust(left=.075,right=.995,bottom=.23,top=.80,wspace=.39)
    specs=[('hidden_rmse_mm',1.,'a  Hidden original motion','Coordinate RMSE (mm)','{:.3f}'),
           ('changed_load_rmse_mm',1.,'b  Response to new loads','Coordinate RMSE (mm)','{:.3f}'),
           ('material_relative_rmse',100.,'c  Material field','Relative RMSE (%)','{:.1f}')]
    rows=[]
    for ax,(key,factor,title,ylabel,fmt) in zip(axes,specs):
        entries=[stats('hierarchical',c,key,factor) for c in conditions]
        bar_panel(ax,entries,CONDITION_COLORS,labels,title,ylabel,fmt)
        rows.append(dict(metric=key,factor=factor,conditions=conditions,statistics=entries))
    fig.text(.5,.965,'Different evidence improves different predictions',ha='center',va='top',fontsize=11,fontweight='bold')
    fig.text(.5,.065,'Views and probes branch independently from the same initial observations',ha='center',fontsize=8.5)
    fig.text(.5,.02,'128 fresh cases  |  Four scalar measurements per branch  |  95% case-bootstrap intervals',ha='center',fontsize=8)
    save(fig,'summary');figuredata['figures']['summary']=rows

    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7));fig.subplots_adjust(left=.075,right=.99,bottom=.26,top=.80,wspace=.43)
    for ax,key,factor,title,ylabel,fmt in [
        (axes[0],'changed_load_rmse_mm',1.,'a  Changed-load prediction','Coordinate RMSE (mm)','{:.3f}'),
        (axes[1],'material_relative_rmse',100.,'b  Material estimation','Relative RMSE (%)','{:.1f}')]:
        entries=[stats(model,'probes',key,factor) for model in MODEL_ORDER]
        bar_panel(ax,entries,MODEL_COLORS,['Fixed','Broader','Hier.','Oracle'],title,ylabel,fmt)
    ax=axes[2];points=[]
    for model,color,marker in zip(MODEL_ORDER,MODEL_COLORS,MODEL_MARKERS):
        x=stats(model,'probes','force_interval_width_N');y=stats(model,'probes','force_coverage90',100)
        ax.errorbar(x['mean'],y['mean'],xerr=np.array([[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]]),
                    yerr=np.array([[y['mean']-y['ci95'][0]],[y['ci95'][1]-y['mean']]]),
                    fmt=marker,color=color,ecolor=color,markersize=6,capsize=2,elinewidth=.9,
                    markeredgecolor='black',markeredgewidth=.5,zorder=4)
        points.append(dict(model=model,width_N=x,coverage_percent=y))
    ax.axhline(90,color='black',linestyle='--',linewidth=.8,zorder=1)
    ax.text(.99,.96,'Nominal 90%',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax.set_title('c  Force uncertainty',loc='left',fontweight='bold',pad=6)
    ax.set_xlabel('Mean 90% interval width (N)',labelpad=3);ax.set_ylabel('Coverage (%)',labelpad=3)
    ax.set_ylim(78,97);clean(ax)
    fig.text(.5,.97,'Force-prior models after the same four calibrated probes',ha='center',va='top',fontsize=11,fontweight='bold')
    handles=[Line2D([0],[0],marker=mark,color='none',markerfacecolor=c,markeredgecolor='black',markeredgewidth=.5,markersize=6,label=label)
             for c,mark,label in zip(MODEL_COLORS,MODEL_MARKERS,['Fixed nominal scale','Variance-matched Gaussian','Hierarchical scale','Oracle true scale'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,frameon=False,fontsize=8.3,columnspacing=2.1,handletextpad=.6)
    save(fig,'prior-models');figuredata['figures']['prior-models']=points

    # Fixed first case of each geometry. Recompute, never select by appearance.
    records=[]
    for shape in ['block','step']:
        data=dict(np.load(ROOT/f'runs/joint-field-surface-v04/{shape}.npz'));m=bank.maps(data)
        truth=dict(np.load(BASE/shape/'truth.npz'));assert str(truth['protocol_sha256'])==sha
        p,_=fit_scale_mixture(m['H'],truth['y'][0],bank.load_cov(),bank.NOISE**2)
        states={'initial':p}
        for condition in ['views','probes']:
            current=p
            for a in SCHEDULES[condition]:current=update(current,m,a,truth['measurements'][0,a],5)
            states[condition]=current
        levels=np.unique(data['theta'][:,0]);truth_x=int(np.argmin(abs(levels-truth['theta'][0,0])));truth_y=int(truth['scale_index'][0])
        for condition in conditions:
            w=states[condition].weights.reshape(-1,5)
            heat=np.stack([w[np.isclose(data['theta'][:,0],level)].sum(0) for level in levels],axis=1)
            assert abs(heat.sum()-1)<1e-10
            record=next(r for r in docs[0 if shape=='block' else 64]['rows'] if r['model']=='hierarchical' and r['condition']==condition)
            records.append(dict(shape=shape,case=0,condition=condition,posterior_mass=heat.tolist(),
                material_scale_kPa=(np.exp(levels)/1000).tolist(),force_scales=SCALES.tolist(),
                true_cell=[truth_x,truth_y],correlation=record['logE_logtau_correlation']))
    vmax=max(np.max(r['posterior_mass']) for r in records)
    cmap=LinearSegmentedColormap.from_list('posterior',['#FFFFFF','#C3E2DF','#5CB1AC','#176964'])
    fig=plt.figure(figsize=(7.2,4.2));grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,.055],left=.075,right=.945,bottom=.17,top=.88,wspace=.31,hspace=.28)
    for i,r in enumerate(records):
        row=i//3;col=i%3;ax=fig.add_subplot(grid[row,col])
        im=ax.imshow(r['posterior_mass'],origin='lower',cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest',aspect='auto')
        truth=ax.plot(*r['true_cell'],marker='x',markersize=7,markeredgewidth=1.5,color='black')[0]
        truth.set_path_effects([patheffects.Stroke(linewidth=3,foreground='white'),patheffects.Normal()])
        ax.set_xticks(range(5),[f'{v:g}' if abs(v-round(v))<1e-8 else f'{v:.1f}' for v in r['material_scale_kPa']])
        ax.set_yticks(range(5),['0.50','0.71','1.00','1.41','2.00'])
        if row==0:ax.set_title(labels[col],fontsize=10,fontweight='bold',pad=5)
        else:ax.set_xlabel('Material scale (kPa)',labelpad=3)
        if col==0:ax.set_ylabel(('Block' if row==0 else 'Stepped block')+'\nForce prior scale',labelpad=4)
        else:ax.set_yticklabels([])
        ax.text(.98,.96,f"Correlation {r['correlation']:.2f}",transform=ax.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(facecolor='white',alpha=.87,edgecolor='none',pad=1.5))
        ax.tick_params(length=2,width=.6,pad=2)
    cax=fig.add_subplot(grid[:,3]);cb=fig.colorbar(im,cax=cax);cb.set_label('Posterior probability per cell',fontsize=9,labelpad=5);cb.ax.tick_params(labelsize=8,length=2)
    fig.text(.5,.98,'Case 0: joint material and force-scale uncertainty',ha='center',va='top',fontsize=11,fontweight='bold')
    fig.text(.5,.065,'Black cross = truth. Shared probability scale; finite 5 x 5 marginal, not a continuous density.',ha='center',fontsize=8)
    fig.text(.5,.025,'Views and probes are separate updates of the initial posterior. No case selection by outcome.',ha='center',fontsize=8)
    save(fig,'case0-joint-posterior');figuredata['figures']['case0-joint-posterior']=records
    figuredata['source_sha256']={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'figure-data.json').write_text(json.dumps(figuredata,indent=2)+'\n');verify_protocol();print('Saved',OUT,flush=True)


if __name__=='__main__':main()
