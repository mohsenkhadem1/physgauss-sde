"""Data-driven stem figures/video; no deformation magnification or image synthesis."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scripts.gap_first_v01.stem_mechanics import displacement,response_row
from scripts.gap_first_v01.stem_inference import plan,action_menu

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/gap-first-v01/stem'
ARIAL=Path('/home/bayespc2/.local/share/fonts/arial/Arial.TTF')
for font_path in ARIAL.parent.glob('*.TTF'):font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family':'Arial','font.size':12,'text.color':'black',
    'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
    'pdf.fonttype':42,'ps.fonttype':42})
COLORS=['#167c94','#bd6741']


def cuboid(ax,x,y,z,dx,dy,dz,color,alpha=1):
    v=np.array([[x+i*dx,y+j*dy,z+k*dz] for k in [0,1] for j in [0,1] for i in [0,1]])
    ix=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    ax.add_collection3d(Poly3DCollection([v[a] for a in ix],facecolors=color,
        edgecolors=(0,0,0,.22),linewidths=.4,alpha=alpha))


def stem(ax,theta,load_fraction=1,force=.02,length=.12,color='#167c94',alpha=1):
    z=np.linspace(0,length,61)
    u=displacement(z,load_fraction*length,length,force,theta)
    # Cross-section rotation from the same mechanical centerline, no exaggeration.
    slope=np.gradient(u,z); normal=np.stack((np.ones_like(slope),np.zeros_like(slope),-slope),1)
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    rings=np.stack([np.stack((u,np.zeros_like(u),z+.012),1)+
        normal*sx*.004+np.array([0,sy*.005,0]) for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]],1)
    faces=[rings[0],rings[-1]]
    for i in range(len(z)-1):
        for j in range(4):faces.append(np.array([rings[i,j],rings[i,(j+1)%4],rings[i+1,(j+1)%4],rings[i+1,j]]))
    ax.add_collection3d(Poly3DCollection(faces,facecolors=color,edgecolors=(0,0,0,.16),
        linewidths=.25,alpha=alpha))
    if alpha>=.9:
        cuboid(ax,u[0]-.009,-.009,.006,.018,.018,.006,'#85949c')
        zz=np.linspace(-.035,u[0]-.009,35)
        yy=.002*np.sin(np.linspace(0,10*np.pi,35))
        ax.plot(zz,yy,np.full_like(zz,.009),color='#404040',lw=1.7)
        cuboid(ax,-.04,-.01,0,.004,.02,.016,'#6e747a')
        cuboid(ax,-.042,-.019,-.003,.088,.038,.003,'#d6dadd')
        at=float(displacement(load_fraction*length,load_fraction*length,length,force,theta))
        ax.quiver(at-.023,0,load_fraction*length+.012,.020,0,0,color='black',
                  arrow_length_ratio=.22,linewidth=1.8)
        ax.scatter([at],[0],[load_fraction*length+.012],s=28,c='black',depthshade=False)
    return u


def setup(ax,azim=-85):
    ax.set_xlim(-.043,.045);ax.set_ylim(-.028,.028);ax.set_zlim(-.004,.143)
    ax.set_box_aspect((.088,.056,.147));ax.view_init(18,azim)
    ax.set_axis_off();ax.set_proj_type('ortho')


def example_figure():
    fig=plt.figure(figsize=(11,3.6),facecolor='white')
    specs=[('Initial tip push',1),('New halfway push',.5),('Held-out third location',.75)]
    for j,(title,a) in enumerate(specs):
        for i,theta in enumerate([[.4,.1],[.1,.4]]):
            ax=fig.add_axes([.02+j*.325+i*.15,.16,.18,.68],projection='3d')
            stem(ax,theta,a,color=COLORS[i]);setup(ax)
            value=float(displacement(a*.12,a*.12,.12,.02,theta))*1000
            ax.text2D(.5,.0,f'{"A" if i==0 else "B"}: {value:.2f} mm',transform=ax.transAxes,ha='center',fontsize=12)
        fig.text(.18+j*.325,.89,title,ha='center',fontsize=14,fontweight='bold')
    fig.text(.5,.985,'Same observed tip displacement. Different response to a new interaction.',
             ha='center',va='top',fontsize=16,fontweight='bold')
    fig.text(.5,.08,'A: 80% bending / 20% base motion     B: 20% bending / 80% base motion',ha='center',fontsize=12)
    fig.text(.5,.025,'20 mN lateral load; actual scale. Full stems shown for explanation; initial inference observes only the tip.',
             ha='center',fontsize=10)
    fig.savefig(OUT/'stem-mechanical-ambiguity.png',dpi=200)
    fig.savefig(OUT/'stem-mechanical-ambiguity.pdf')
    plt.close(fig)


def results_figures():
    summary=json.loads((ROOT/'reports/gap-first-v01/stem/summary.json').read_text())
    rows=json.loads((ROOT/'reports/gap-first-v01/stem/outcomes.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    policies=['proceed','fixed_look','fixed_half_push','covariance_gaussian','risk_policy']
    labels=['No action','Always look','Always push','Gaussian risk','Full posterior']
    for ax,cond,title in zip(axes,['repeat_tip','visible_base','concealed_base'],
        ['Repeat tip: proceed','Base visible: look','Base hidden: push']):
        rr=summary['conditions'][cond]
        vals=[rr[p]['models']['posterior_mean']['rmse_mm'] for p in policies]
        ax.bar(np.arange(len(vals)),vals,color=['#aebac0']*4+['#167c94'])
        ax.set_xticks(np.arange(len(vals)),labels,rotation=32,ha='right',fontsize=9)
        ax.set_title(title,fontsize=12);ax.set_ylabel('Held-out response RMSE (mm)')
        ax.spines[['top','right']].set_visible(False)
        for j,v in enumerate(vals):ax.text(j,v+.018,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(vals)*1.22+.07)
    fig.savefig(OUT/'stem-response-comparison.png',dpi=200);fig.savefig(OUT/'stem-response-comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for ax,cond,title in zip(axes,['repeat_tip','visible_base','concealed_base'],['Repeat tip','Base visible','Base hidden']):
        rr=[r for r in rows if r['condition']==cond]
        risk=np.array([r['predictions']['posterior_mean_risk'] for r in rr])
        err=np.array([abs(r['predictions']['posterior_mean']-r['true_response_m'])*1000 for r in rr])
        ax.scatter(risk,err,s=14,alpha=.65,c='#167c94',edgecolors='none')
        ax.axhline(.5,color='black',linestyle='--',lw=1,label='0.5 mm tolerance')
        ax.set(xlim=(-.03,1.03),xlabel='Risk recorded before outcome',ylabel='Realized error (mm)',title=title)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'stem-pre-outcome-risk.png',dpi=200);fig.savefig(OUT/'stem-pre-outcome-risk.pdf');plt.close(fig)
    sweep=json.loads((ROOT/'reports/gap-first-v01/stem/visibility-sweep.json').read_text())
    fig,ax=plt.subplots(figsize=(5.2,3.0),layout='constrained')
    visible=np.array([1-r['lowest_visible_fraction'] for r in sweep['rows']])
    risk=np.array([r['expected_failure'] for r in sweep['rows']])
    ax.plot(visible*100,risk,'o-',color='#167c94',lw=2,ms=4)
    ax.axhline(.10,color='black',ls='--',lw=1)
    ax.set(xlabel='Stem revealed below tip (%)',ylabel='Expected failure after a new view',
           ylim=(-.015,.85),xlim=(-1,101))
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'stem-visibility.png',dpi=200);fig.savefig(OUT/'stem-visibility.pdf');plt.close(fig)


def posterior_figure():
    c=json.loads((ROOT/'scripts/gap_first_v01/stem_protocol.json').read_text())
    p,row,decision=plan(.01,'concealed_base',.12,c)
    action=next(a for a in action_menu('concealed_base',.12,c) if a['name']==decision['selected_action'])
    truth=np.array([.4,.1]);post=p.update(action['row'],float(truth@action['row']),action['sigma'])
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
    for ax,dist,title in zip(axes[:2],[p,post],['After tip observation','After calibrated push']):
        z=dist.weights.reshape(c['grid_size'],c['grid_size']).T
        ax.imshow(z,origin='lower',extent=[.04,.48,.04,.48],cmap='Blues',aspect='equal',vmin=0,vmax=z.max())
        ax.scatter([.4,.1],[.1,.4],c=['black','white'],edgecolor='black',s=30)
        ax.text(.41,.08,'A',fontsize=11);ax.text(.11,.41,'B',fontsize=11)
        ax.set(xlabel='Bending compliance (m/N)',ylabel='Base compliance (m/N)',title=title)
    tasks=[response_row(a*.12,a*.12,.12,.02) for a in [1,.75]]
    before=[p.predictions(r,.0005)['posterior_mean_risk'] for r in tasks]
    after=[post.predictions(r,.0005)['posterior_mean_risk'] for r in tasks]
    x=np.arange(2);ax=axes[2]
    ax.bar(x-.16,before,width=.32,color='#aebac0',label='Before probe')
    ax.bar(x+.16,after,width=.32,color='#167c94',label='After probe')
    ax.set_xticks(x,['Repeat tip','New load at 75%']);ax.set_ylim(0,1)
    ax.set_ylabel('Predicted failure probability');ax.set_title('Risk depends on the next action')
    ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'stem-posterior-and-task-risk.png',dpi=200);fig.savefig(OUT/'stem-posterior-and-task-risk.pdf');plt.close(fig)


def movie(frames=180):
    c=json.loads((ROOT/'scripts/gap_first_v01/stem_protocol.json').read_text())
    p,row,decision=plan(.01,'concealed_base',.12,c)
    action=next(a for a in action_menu('concealed_base',.12,c) if a['name']==decision['selected_action'])
    theta=np.array([.4,.1]);before=p.weights@p.theta
    # Noiseless illustrative measurement, separate from noisy sealed cohort.
    post=p.update(action['row'],float(theta@action['row']),action['sigma']);after=post.weights@post.theta
    fig=plt.figure(figsize=(12,5.6),facecolor='white')
    axs=[fig.add_axes([.015+i*.325,.11,.32,.73],projection='3d') for i in range(3)]
    closeups=[fig.add_axes([.035+i*.325,.58,.105,.18]) for i in range(3)]
    heading=fig.text(.5,.96,'',ha='center',fontsize=17,fontweight='bold')
    labels=['Actual specimen A','Prediction before calibration','Prediction after calibration']
    for i,label in enumerate(labels):fig.text(.175+i*.325,.88,label,ha='center',fontsize=13,fontweight='bold')
    info=fig.text(.5,.07,'',ha='center',fontsize=12)
    fig.text(.5,.042,'Close-ups: solid line = actual; dashed line = prediction.',ha='center',fontsize=9)
    fig.text(.5,.018,'Analytical quasistatic replay; no deformation magnification. Full geometry shown; input is a tip track. Calibration example uses noiseless evidence.',
             ha='center',fontsize=9)
    matplotlib.rcParams['animation.ffmpeg_path']=str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg')
    writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','19'])
    with writer.saving(fig,str(OUT/'stem-actual-versus-predicted.mp4'),dpi=120):
        for frame in range(frames):
            phase=min(2,int(frame/(frames/3)));t=(frame%(frames/3))/(frames/3-1)
            ramp=min(1,t/.35) if t<.8 else max(0,(1-t)/.2)
            frac=[1,action['location_fraction'],.75][phase];force=.02*ramp
            heading.set_text(['Initial tip push: same observation, different explanations',
                f'Calibrated push at {frac:.0%} of stem height',
                'New interaction at 75% height: withheld from calibration'][phase])
            for i,(ax,param) in enumerate(zip(axs,[theta,before,after])):
                ax.clear()
                inset=closeups[i];inset.clear()
                if i==2 and phase<2:
                    setup(ax,-70)
                    ax.text2D(.5,.5,'Awaiting calibration',transform=ax.transAxes,ha='center',fontsize=12)
                    inset.set_axis_off()
                    continue
                stem(ax,param,frac,force,color=COLORS[0] if i==0 else '#7a9d64')
                setup(ax,-70+12*np.sin(2*np.pi*frame/frames))
                value=float(displacement(frac*.12,frac*.12,.12,force,param))*1000
                ax.text2D(.5,.02,f'Displacement: {value:.2f} mm',transform=ax.transAxes,ha='center',fontsize=12)
                # A literal zoom of centerline coordinates, not stretched motion.
                z=np.linspace(max(0,frac*.12-.008),min(.12,frac*.12+.008),30)
                real=displacement(z,frac*.12,.12,force,theta)*1000
                estimate=displacement(z,frac*.12,.12,force,param)*1000
                inset.plot(real,(z-frac*.12)*1000,color=COLORS[0],lw=2)
                if i:inset.plot(estimate,(z-frac*.12)*1000,color='#7a9d64',lw=2,ls='--')
                inset.set(xlim=(-1,12),ylim=(-8,8),xticks=[0,5,10],yticks=[-8,0,8])
                inset.set_aspect('equal');inset.tick_params(labelsize=7,length=2)
                inset.set_title('Centerline close-up',fontsize=8)
                inset.set_xlabel('Displacement (mm)',fontsize=7,labelpad=1)
                inset.spines[['top','right']].set_visible(False)
            info.set_text(f'Applied force: {force*1000:.1f} mN     |     Before-calibration risk for held-out task: {decision["predictions"]["posterior_mean_risk"]:.0%}')
            writer.grab_frame()
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',action='store_true');ap.add_argument('--video',action='store_true')
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    example_figure()
    if args.results:results_figures();posterior_figure()
    if args.video:movie()
