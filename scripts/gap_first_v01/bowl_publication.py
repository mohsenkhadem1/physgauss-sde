"""Figures and factual TeX section from sealed bowl results, fixed case0 only."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .bowl_campaign import ROOT,REPORT,INFERENCE_N,REFERENCE_N,THETA
from .bowl_visualize import compare
from .quantiles import gaussian_mixture_quantiles

OUT=Path('output/gap-first-v01/bowl')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42

def trajectory(index,mixture=False):
    n=REFERENCE_N if mixture and index>=len(THETA) else INFERENCE_N
    return ROOT/f'fem-n{n}'/f'h{index%len(THETA):02d}-heldout_push.npz'

def get(rows,cohort,model,stage,h=0,rep=0):
    return next(r for r in rows if r['cohort']==cohort and r['model']==model and r['stage']==stage and r['hypothesis']==h and r['replicate']==rep)

def force_figure(rows):
    truth_path=ROOT/f'fem-n{REFERENCE_N}'/'h00-heldout_push.npz'
    with np.load(truth_path) as a:truth=a['reaction_averages'];times=a['times']
    fig,axes=plt.subplots(2,2,figsize=(9,5.5),sharex=True)
    for col,model in enumerate(('fem','resolution-mixture')):
        mixture=model=='resolution-mixture';count=32 if mixture else 16
        bank=[]
        for h in range(count):
            with np.load(trajectory(h,mixture)) as a:bank.append(a['reaction_averages'])
        bank=np.asarray(bank)
        for row,(body,axis,label) in enumerate([(0,2,'Bowl reaction Fz (N)'),(1,1,'Tool reaction Fy (N)')]):
            ax=axes[row,col];ax.plot(times,truth[:,body,axis],color='black',lw=1.8,label='Reference')
            for stage,name,color in [('initial','Before calibration','#ce8136'),('active','After calibration','#168b87')]:
                record=get(rows,'mesh-mismatch',model,stage);w=np.asarray(record['weights']);values=bank[:,:,body,axis]
                lo,hi=gaussian_mixture_quantiles(w,values,np.zeros_like(values),(.05,.95))
                ax.plot(times,w@values,color=color,lw=1.2,label=name);ax.fill_between(times,lo,hi,color=color,alpha=.17,linewidth=0)
            ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
            if row==0:ax.set_title('Coarse-only posterior' if col==0 else 'Resolution-mixture control')
            else:ax.set_xlabel('Physical time (s)')
    axes[0,0].legend(frameon=False,fontsize=9.5,loc='lower left')
    fig.text(.5,.012,'Fixed world 0, finer reference. Bands: conditional 90% coordinate intervals. 20 ms trailing boxcar averages.',ha='center',fontsize=9.5)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'bowl-force-predictions.png',dpi=220)
    fig.savefig(OUT/'bowl-force-predictions.pdf');plt.close(fig)

def all_worlds_figure(rows):
    fig,axes=plt.subplots(2,2,figsize=(8,6.5))
    palette=['#28779d','#36a6a0','#b78e36','#a5595a']
    metrics=[('map_rms_mm','Motion prediction RMS (mm)'),('tool_reaction_map_vector_rms_N','Tool-force vector RMS (N)')]
    records=[]
    for col,cohort in enumerate(('matched','mesh-mismatch')):
        for row,(metric,label) in enumerate(metrics):
            ax=axes[row,col];points=[]
            for h in range(16):
                x=np.mean([get(rows,cohort,'fem','initial',h,rep)[metric] for rep in range(2)])
                y=np.mean([get(rows,cohort,'fem','active',h,rep)[metric] for rep in range(2)])
                points.append((x,y));ax.scatter(x,y,s=45,color=palette[h//4],edgecolor='black',linewidth=.4)
                records.append(dict(cohort=cohort,metric=metric,world=h,before=float(x),after=float(y)))
            limit=max(max(max(p) for p in points)*1.1,.1)
            ax.plot([0,limit],[0,limit],color='black',lw=.8,ls='--');ax.set(xlim=(0,limit),ylim=(0,limit))
            ax.set_xlabel('Before calibration');ax.set_ylabel('After calibration')
            ax.set_title(('Matched finite control' if col==0 else 'Finer space/time stress')+'\n'+label)
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',linestyle='',markerfacecolor=c,markeredgecolor='black',label=f'{e:g} kPa') for c,e in zip(palette,[2.5,4,6.5,10])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=4,frameon=False,title="Young's modulus",fontsize=10,title_fontsize=10)
    fig.text(.5,.012,'All 16 worlds; two noise replicates per point. Below diagonal: lower error after calibration.',ha='center',fontsize=9.5)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(OUT/'bowl-all-worlds.png',dpi=220)
    fig.savefig(OUT/'bowl-all-worlds.pdf');plt.close(fig)
    (OUT/'bowl-all-worlds-data.json').write_text(json.dumps(records,indent=2))

def section(summary):
    def val(model,cohort,stage,key):return summary['groups'][f'{model}/{cohort}/{stage}'][key]['mean']
    lines=[]
    for cohort,label in [('matched','Matched'),('mesh-mismatch','Finer reference')]:
        for model,name in [('fem','Coarse only'),('resolution-mixture','Resolution mixture')]:
            before=val(model,cohort,'initial','map_rms_mm');after=val(model,cohort,'active','map_rms_mm')
            bowl=val(model,cohort,'active','bowl_reaction_map_vector_rms_N');tool=val(model,cohort,'active','tool_reaction_map_vector_rms_N')
            brier=val(model,cohort,'active','brier')
            lines.append(f'{label} & {name} & {before:.2f} & {after:.2f} & {bowl:.3f} & {tool:.3f} & {brier:.3f} \\\\')
    qualification=json.loads((REPORT/'numerical-qualification.json').read_text())
    visibility=json.loads((REPORT/'all-state-geometry-audit.json').read_text())
    failures=visibility['total_visibility_failures'];checks=visibility['total_visibility_checks']
    visibility_text=(f'All {checks} declared marker observations pass the surface/tool/bowl occlusion audit.' if failures==0 else f'The audit finds {failures} occluded observations among {checks}; all are retained, so these results must be interpreted as supplied oracle tracks rather than fully camera-visible measurements.')
    text=r'''\section{Nonlinear bowl contact under position control}
\label{sec:gap-first-bowl}
We test a soft, tapered nonbox solid falling under gravity into a known rigid
hemispherical bowl (radius 90\,mm), followed by a position-controlled interaction.
The model uses compressible Neo--Hookean tetrahedral mechanics, material-surface
quadrature, penalty contact and regularized friction. It is a bounded nonlinear
contact approximation, not the exact Gaussian force-elimination model.
Geometry, density, damping and gravity are known; Young's modulus and a shared
bowl/tool friction coefficient are uncertain.

\paragraph{Observation and prediction protocol.}
Sixteen parameter worlds combine four moduli (2.5,4,6.5,10\,kPa) and four friction
coefficients (0.1,0.25,0.45,0.7). We supply four known surface correspondences
projected into an overhead camera, with 0.5\,mm independent coordinate noise.
Initial observations occur at 0.08 and 0.12\,s. Exactly one reset calibration
experiment is selected: a 25\,mm position-controlled push or a 0.2\,rad bowl rotation.
The latter moves the wall tangentially about a fixed spherical center; gravity
does not rotate. New coordinates at 0.94 and 1.2\,s update the posterior. The
separate held-out experiment applies a stronger 45\,mm transverse push after
entering through the open rim. No reaction-force measurements enter inference.

The primary risk is the posterior probability that the deployed MAP model
exceeds 3\,mm point RMS across 64 common material nodes at 1.1,1.2 and 1.3\,s.
This is a fixed-budget calibration comparison, not a proceed-or-acquire policy.
Both fixed actions and the same-evidence MAP predictor are controls. All action
choices and risks are sealed before scoring the selected-world held-out targets.
We evaluate 16 worlds at two discretizations, with two noise replicates each:
64 records, not 64 independent objects. Intervals resample the 16 world clusters.

\paragraph{Qualification and limits.}
Inference uses a 6-cell-per-axis mesh at 25\,$\mu$s; the finer space/time reference
uses 12 cells at 12.5\,$\mu$s, preserving the same piecewise-linear rest geometry.
A 12-cell 25\,$\mu$s development run failed the positive-volume check and is retained
as a numerical failure. The smaller-step soft/stiff controls complete with minimum
Jacobians 0.271/0.372 and sampled penetration 0.172/0.268\,mm. Dense saved-surface
penetration is audited separately. Halving the inference timestep changes the
development target by 0.35\,mm; changing both space and time resolution changes it
by 1.42 and 2.81\,mm. The latter is close to the 3\,mm decision tolerance: this is an
intentional numerical-model stress test, not converged continuum truth.
The open bowl has no rim thickness, continuous collision detection or self-contact.
VISIBILITYTEXT

\paragraph{Conditional prediction and numerical-model uncertainty.}
Table~\ref{tab:bowl-gap-results} reports the predeclared primary coarse-only posterior
and a secondary 32-hypothesis resolution mixture. The mixture assigns prior 0.5 to
each resolution and the same uniform parameter prior within each. The finer
reference is therefore included by construction; this comparator demonstrates
conditional numerical-model uncertainty, not coverage of arbitrary unknown physics.
The complete report also compares both fixed actions and gives the mixture exactly
the coarse policy's chosen observation, separating model uncertainty from action
selection. Reaction-force errors use 20\,ms trailing boxcar averages accumulated
from every integration step, with bowl and tool scored separately in newtons.

\paragraph{Measured benefit and failure.}
In the matched finite control, selected calibration reduces mean MAP motion error
from 7.588 to 0.023\,mm, but fixed bowl rotation achieves zero MAP error; this does
not establish an advantage over the best fixed experiment. Against the finer
space/time reference, calibration reduces mean error from 6.981 to 3.091\,mm,
while 43.75\% of predictions exceed the 3\,mm tolerance despite a mean predicted
failure probability of only 1.856\%. Conditional 90\% coordinate intervals cover
only 6.14\% of reference coordinates. Fixed rotation has a lower 31.25\% failure
rate. Thus improved motion accuracy does not imply trustworthy uncertainty under
model mismatch. Given exactly the coarse policy's correction observations, the
resolution-mixture control reaches 0.051\,mm mean error and no failures, conditional
on the fine model being included by construction.

\begin{table*}[t]
\centering\small
\begin{tabular}{llrrrrr}
\toprule
Reference & Posterior & Initial motion & After motion & Bowl force & Tool force & Brier \\
 & & \multicolumn{2}{c}{MAP RMS (mm)} & \multicolumn{2}{c}{MAP vector RMS (N)} & \\
\midrule
TABLEROWS
\bottomrule
\end{tabular}
\caption{Bowl study, mean over 16 parameter worlds and two noise replicates.
``After'' uses one selected calibration experiment. Finer-reference results are a
space/time-discretization stress; the secondary mixture contains that reference
model by construction. Full clustered intervals, both fixed controls and conditional
motion/force coverage are retained in the report.}
\label{tab:bowl-gap-results}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=.97\textwidth]{../../output/gap-first-v01/bowl/bowl-all-worlds.pdf}
\caption{All 16 parameter worlds in the controlled finite grid, not 16 distinct
objects. Each point averages two observation-noise replicates. Horizontal and
vertical axes give errors before and after selected calibration; points below
the diagonal improve. Colors identify Young's modulus, with four friction values
per modulus. The matched control and finer space/time stress are shown separately;
these plots do not establish superiority over the best fixed calibration action.}
\label{fig:bowl-all-worlds}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=.97\textwidth]{../../output/gap-first-v01/bowl/bowl-refined-case00.png}
\caption{Fixed world 0, noise replicate 0: actual finer-reference geometry and
coarse-only MAP predictions before/after calibration on a separate reset interaction.
All panels use the same camera and true spatial scale; no deformation amplification.
Color encodes position error and saturates at 6\,mm. This is simulated surface
geometry, not a photorealistic-rendering benchmark.}
\label{fig:bowl-fixed-case}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=.9\textwidth]{../../output/gap-first-v01/bowl/bowl-force-predictions.pdf}
\caption{Withheld reaction predictions for fixed world 0 of the finer-reference
stress. Only position tracks are observed. Shading gives conditional 90\% coordinate
intervals; the secondary resolution mixture contains the reference by construction.}
\label{fig:bowl-forces}
\end{figure*}

An unchanged PhysTwin spring-force kernel was also attempted under the shared
contact/integration adapter. It failed that adapter's positive-tetrahedron-volume
qualification across CPU/GPU and timestep checks. We retain the failure record but
exclude it from accuracy rankings; it is not evidence that the full native PhysTwin
method fails on bowls.
'''
    text=text.replace('VISIBILITYTEXT',visibility_text).replace('TABLEROWS','\n'.join(lines))
    # Approved layout trial6: declare related floats before protocol prose.
    pattern=r"\\begin\{(table\*|figure\*)\}.*?\\end\{\1\}"
    floats=re.findall(pattern,text,flags=re.S)
    blocks=[m.group(0) for m in re.finditer(pattern,text,flags=re.S)]
    assert floats==['table*','figure*','figure*','figure*']
    text=re.sub(pattern,'',text,flags=re.S)
    widths={'bowl-all-worlds.pdf':'.82','bowl-refined-case00.png':'.75','bowl-force-predictions.pdf':'.75'}
    for i,block in enumerate(blocks):
        for filename,width in widths.items():
            if filename in block:
                block=re.sub(r'width=\.[0-9]+(?=\\textwidth)',lambda _: 'width='+width,block)
        blocks[i]=block
    marker=r'\paragraph{Observation and prediction protocol.}'
    assert text.count(marker)==1
    text=text.replace(marker,'\n'.join(blocks)+'\n'+marker)
    path=Path('papers/cvpr2027-joint-identification/sections/gap-first-bowl.tex');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((REPORT/'case-results.json').read_text());summary=json.loads((REPORT/'summary.json').read_text())
    for cohort,n in [('matched',INFERENCE_N),('mesh-mismatch',REFERENCE_N)]:
        initial=get(rows,cohort,'fem','initial');active=get(rows,cohort,'fem','active')
        prefix='bowl-matched-case00' if cohort=='matched' else 'bowl-refined-case00'
        compare(ROOT/f'fem-n{n}'/'h00-heldout_push.npz',trajectory(initial['map_index']),trajectory(active['map_index']),OUT/prefix,
                labels=['Reference','Before calibration (MAP)','After calibration (MAP)'])
    force_figure(rows);all_worlds_figure(rows);section(summary)
    (OUT/'publication-manifest.json').write_text(json.dumps(dict(case_selection='world 0,noise replicate 0 fixed before outcomes; all 16 worlds also plotted',source_results_sha256=hashlib.sha256((REPORT/'case-results.json').read_bytes()).hexdigest(),font='Arial, black text',geometry='actual saved states, no deformation amplification',secondary_mixture='Fine reference included by construction; no unknown-model calibration claim'),indent=2))

if __name__=='__main__':main()
