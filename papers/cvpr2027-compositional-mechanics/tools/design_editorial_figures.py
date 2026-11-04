"""Second visual design: projected geometry, scene cards, and direct comparisons.

Uses only the same sealed world00/query0/final-frame examples and full-cohort
scores as design_visual_paper.py. Illustration geometry is explicitly schematic.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import design_visual_paper as d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Circle
import json

INK='#183044'; MUTED='#65798a'; TEAL='#008c88'; GOLD='#dc8032'; PURPLE='#8766ac'
AZ=np.deg2rad(-64); EL=np.deg2rad(24)
R=np.array([[-np.sin(AZ),np.cos(AZ),0],[-np.sin(EL)*np.cos(AZ),-np.sin(EL)*np.sin(AZ),np.cos(EL)],[np.cos(EL)*np.cos(AZ),np.cos(EL)*np.sin(AZ),np.sin(EL)]])
def project(x):return x@R.T
def extent(items):
    p=project(np.concatenate([x.reshape(-1,3) for x in items]));lo=p[:,:2].min(0);hi=p[:,:2].max(0)
    pad=max(hi-lo)*.055
    return lo-pad,hi+pad
def mesh(ax,x,limits,error=None,base=TEAL,alpha=1):
    p=project(x);tris=p[d.FACES]; order=np.argsort(tris[:,:,2].mean(1))
    if error is None:
        polys=x[d.FACES];n=np.cross(polys[:,1]-polys[:,0],polys[:,2]-polys[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
        light=.70+.30*np.maximum(n@np.array([.25,-.6,.75]),0)
        colors=np.clip(np.array(d.to_rgb(base))[None,:]*light[:,None],0,1)
    else:colors=d.CMAP(d.NORM(error[d.FACES].mean(1)))
    ax.add_collection(PolyCollection(tris[order,:,:2],facecolors=colors[order],edgecolors=(.12,.22,.3,.035),linewidths=.055,alpha=alpha,rasterized=True))
    lo,hi=limits;ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.axis('off')
def obj(fig,rect,x,limits,error=None,base=TEAL):
    ax=fig.add_axes(rect);ax.patch.set_alpha(0)
    for k,b in enumerate(x):mesh(ax,b,limits,error=None if error is None else error[k],base=base if k==0 else PURPLE)
    return ax
def text(fig,x,y,s,size=11,color=INK,weight='normal',ha='left',**kw):fig.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,**kw)
def arrow(ax,start,end,color=MUTED,rad=0,lw=1.5):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}'))
def block():
    x=d.Q*np.array([1.8,.75,.75]);x[:,1]-=.375;x[:,2]-=.375;x[:,2]+=.22*np.sin(np.pi*d.Q[:,0]);return x

def method():
    fig=plt.figure(figsize=(12,4.05));canvas=fig.add_axes([0,0,1,1]);canvas.axis('off')
    # A composition, rather than a row of generic cards.
    canvas.add_patch(FancyBboxPatch((.29,.13),.43,.73,boxstyle='round,pad=.01,rounding_size=.025',facecolor='#f1f7f7',edgecolor='#cfe1df',lw=1))
    text(fig,.018,.925,'01  OBSERVE',11,d.BLUE,'bold');text(fig,.305,.925,'02  COMPOSE LOCAL MECHANICS',11,TEAL,'bold');text(fig,.76,.925,'03  PREDICT',11,GOLD,'bold')
    x=block();lim=extent([x])
    for i in range(3):
        xx=x.copy();xx[:,2]+=.06*i*np.sin(np.pi*d.Q[:,0])
        a=obj(fig,[.018+i*.033,.51-i*.075,.17,.27],xx[None],extent([xx]),base=['#a9c9e3','#6ba5d0',d.BLUE][i])
        p=project(xx[::65]);a.scatter(p[:,0],p[:,1],s=10,color=GOLD,zorder=10)
    text(fig,.018,.26,'A short position-track prefix',10,INK,'bold')
    text(fig,.018,.17,'Known geometry + mass\nFit material / attachment descriptors',9,MUTED,linespacing=1.5)
    arrow(canvas,(.235,.50),(.292,.50),TEAL)
    # Material-carried state and a separate boundary graph.
    obj(fig,[.31,.36,.225,.34],x[None],lim,error=(d.Q[:,0]*11)[None])
    text(fig,.315,.765,'Material that moves',11,TEAL,'bold')
    text(fig,.315,.70,r'$z_m(X)\;\longrightarrow\;W_\theta(F,z_m)$',11)
    text(fig,.315,.325,'Spatial material field',9,MUTED)
    # Learned decoder glyph: two input descriptors, hidden units, energy basis.
    for xx,ys in [(.56,[.59,.69]),(.605,[.54,.60,.66,.72]),(.65,[.57,.63,.69])]:
        for yy in ys:canvas.add_patch(Circle((xx,yy),.007,facecolor=TEAL,edgecolor='white',lw=.7,zorder=4))
    for x1,ys1,x2,ys2 in [(.56,[.59,.69],.605,[.54,.60,.66,.72]),(.605,[.54,.60,.66,.72],.65,[.57,.63,.69])]:
        for y1 in ys1:
            for y2 in ys2:canvas.plot([x1,x2],[y1,y2],color='#9dbfbd',lw=.8,zorder=3)
    text(fig,.55,.765,'Frozen decoders',10,TEAL,'bold')
    text(fig,.55,.45,'Energy → stress\nQ2 mechanics → motion',10,INK,linespacing=1.5)
    canvas.plot([.31,.695],[.285,.285],color='#cfdfdf',lw=1)
    text(fig,.315,.22,'INTERACTIONS',8,GOLD,'bold')
    text(fig,.43,.22,'Tool poses  •  attachments  •  contact  •  gravity',9)
    text(fig,.315,.16,r'$\Pi = \int W_\theta\,dV + \int\psi_\phi\,dA + \Pi_{contact} - f_g^\top u$',11)
    arrow(canvas,(.72,.53),(.765,.68),TEAL);arrow(canvas,(.72,.50),(.765,.30),GOLD)
    out=x.copy();out[:,2]+=.18*d.Q[:,0]**2
    obj(fig,[.77,.55,.21,.27],out[None],extent([out]),base=TEAL)
    text(fig,.77,.50,'New deformation + reactions',10,TEAL,'bold')
    a=fig.add_axes([.78,.19,.20,.24]);a.axis('off');p=project(out);ids=np.arange(0,len(p),5)
    for k in ids[np.argsort(p[ids,2])]:
        a.add_patch(Ellipse(p[k,:2],.10,.052,angle=-18,facecolor=plt.cm.viridis(d.Q[k,0]),edgecolor='none',alpha=.48))
    lo,hi=extent([out]);a.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
    text(fig,.77,.12,'Transport Gaussian appearance',10,GOLD,'bold')
    text(fig,.02,.035,'Scene fitting changes descriptors, not shared network weights. Illustrations are schematic; physical and image errors are evaluated separately.',9,MUTED)
    canvas.set(xlim=(0,1),ylim=(0,1));d.save(fig,'method_visual')

def results():
    fig=plt.figure(figsize=(12,6.4));can=fig.add_axes([0,0,1,1]);can.axis('off')
    text(fig,.015,.967,'PHYSICAL GENERALIZATION',12,INK,'bold')
    text(fig,.48,.967,'24 worlds  /  48 held interactions  /  6 scene families',10,MUTED)
    # Every card shows the same preselected query, with shared within-card bounds.
    for i,(family,label) in enumerate(zip(d.FAMILIES,d.LABELS)):
        col=i%3;row=i//3;left=.012+col*.332;bottom=.205+(1-row)*.355
        can.add_patch(FancyBboxPatch((left,bottom),.316,.333,boxstyle='round,pad=.004,rounding_size=.012',facecolor='#f7f9fb',edgecolor='#e1e7ec',lw=.8))
        text(fig,left+.012,bottom+.300,f'{chr(97+i)}  {label}',11,INK,'bold')
        items,t=d.selected(family);keys=['reference','ours','phystwin'];lim=extent([items[m][-1] for m in keys])
        for j,m in enumerate(keys):
            xx=items[m][-1];err=np.linalg.norm(xx-items['reference'][-1],axis=-1)*1000
            obj(fig,[left+.007+j*.103,bottom+.115,.100,.160],xx,lim,error=None if m=='reference' else err,base=d.BLUE)
            text(fig,left+.057+j*.103,bottom+.096,{'reference':'Reference','ours':'Ours','phystwin':'PhysTwin'}[m],8.5,d.COLORS.get(m,MUTED),'bold' if m=='ours' else 'normal',ha='center')
        a=fig.add_axes([left+.040,bottom+.035,.251,.030]);a.set_xlim(0,30);a.set_ylim(-1,1);a.set_yticks([])
        a.set_xticks([0,10,20,30]);a.tick_params(axis='x',labelsize=7,length=2,pad=1);a.spines[['left','top','right']].set_visible(False);a.spines['bottom'].set_color('#cad5df')
        vals={m:np.mean([r['metrics']['vector_rms_mm'] for r in d.DATA if r['method']==m and r['family']==family]) for m in d.COLORS}
        a.plot([min(vals.values()),max(vals.values())],[0,0],color='#b9c8d2',lw=2,zorder=0)
        for m in ['analytic-control','spring-gaus','phystwin','ours']:
            a.scatter(vals[m],{'analytic-control':-.6,'spring-gaus':-.2,'phystwin':.2,'ours':.6}[m],s=49 if m=='ours' else 30,marker='D' if m=='analytic-control' else 'o',c=d.COLORS[m],edgecolors='white',linewidths=.8,zorder=3 if m=='ours' else 2)
        reduction=100*(1-vals['ours']/vals['phystwin'])
        text(fig,left+.305,bottom+.299,f'−{reduction:.0f}%',11,TEAL,'bold',ha='right')
    for i,m in enumerate(d.COLORS):
        x=.026+i*.15;can.scatter(x,.155,s=30,c=d.COLORS[m],marker='D' if m=='analytic-control' else 'o');text(fig,x+.012,.150,d.NAMES[m],8.5)
    text(fig,.025,.115,'Dots: mean motion RMS (mm), 8 queries/family. Percentages: reduction versus PhysTwin.',8.5,MUTED)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=d.NORM,cmap=d.CMAP),cax=fig.add_axes([.695,.147,.28,.012]),orientation='horizontal',extend='max');cb.ax.tick_params(labelsize=7,length=2,pad=1);cb.set_label('Surface position error (mm); >20 saturates',fontsize=8,labelpad=1)
    for i,m in enumerate(['ours','phystwin','spring-gaus','analytic-control']):
        v=np.mean([r['metrics']['vector_rms_mm'] for r in d.DATA if r['method']==m and r['state']=='complete'])
        x=.024+i*.246;text(fig,x,.038,f'{v:.2f}',22,d.COLORS[m],'bold');text(fig,x+.103,.058,d.NAMES[m],9,d.COLORS[m],'bold');text(fig,x+.103,.028,'mm · all 48 queries',8,MUTED)
    can.set(xlim=(0,1),ylim=(0,1));d.save(fig,'held_visual_results')

def plate():
    fig=plt.figure(figsize=(11.6,10));can=fig.add_axes([0,0,1,1]);can.axis('off')
    methods=['reference','ours','analytic-control','phystwin','spring-gaus']; records=[]
    for j,m in enumerate(methods):text(fig,.22+j*.17,.963,'Reference' if m=='reference' else d.NAMES[m],11,d.COLORS.get(m,INK),'bold',ha='center')
    for i,(f,label) in enumerate(zip(d.FAMILIES,d.LABELS)):
        items,t=d.selected(f); y=.79-i*.128;lim=extent([x[-1] for x in items.values()]);errors={}
        if i%2==0:can.add_patch(FancyBboxPatch((.005,y-.012),.99,.125,boxstyle='round,pad=.003',facecolor='#f4f7f9',edgecolor='none'))
        text(fig,.013,y+.066,label.replace(' ','\n',1),10,INK,'bold');text(fig,.013,y+.014,f'{t[-1]:.3f} s',8,MUTED)
        for j,m in enumerate(methods):
            x=items[m][-1];err=np.linalg.norm(x-items['reference'][-1],axis=-1)*1000
            obj(fig,[.145+j*.17,y+.017,.15,.092],x,lim,error=None if m=='reference' else err,base=d.BLUE)
            if m!='reference':
                rms=float(np.sqrt(np.mean(err**2)));errors[m]=rms;text(fig,.22+j*.17,y+.001,f'{rms:.2f} mm',9,d.COLORS[m],'bold' if m=='ours' else 'normal',ha='center')
        records.append(dict(family=f,time_s=float(t[-1]),frame_vector_rms_mm=errors))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=d.NORM,cmap=d.CMAP),cax=fig.add_axes([.30,.07,.47,.015]),orientation='horizontal',extend='max');cb.set_label('Point-position error (mm) · common scale · values above 20 mm saturate',fontsize=9)
    text(fig,.5,.015,'All six families · first predefined world · query 0 · final frame · shared view and scale within each row',9,MUTED,ha='center')
    can.set(xlim=(0,1),ylim=(0,1));d.save(fig,'held_error_plate');return records

def ambiguity():
    fig=plt.figure(figsize=(7.2,1.9));can=fig.add_axes([0,0,1,1]);can.axis('off')
    x=block();lim=extent([x])
    for left,title in [(.02,'Material scale 1×'),(.34,'Material scale 2×')]:
        text(fig,left+.13,.86,title,10,INK,'bold',ha='center')
        obj(fig,[left,.26,.26,.50],x[None],lim,base=TEAL)
    text(fig,.315,.50,'=',24,MUTED,ha='center')
    text(fig,.315,.10,'Identical shape in every camera',9,TEAL,'bold',ha='center')
    a=fig.add_axes([.78,.27,.19,.48]);a.barh([1,0],[1,2],height=.48,color=[d.BLUE,GOLD]);a.set_xlim(0,2.5);a.set_xticks([0,1,2]);a.tick_params(labelsize=8);a.set_yticks([1,0],['1×','2×'],fontsize=8);a.spines[['left','top','right']].set_visible(False)
    text(fig,.86,.86,'Reaction force',10,INK,'bold',ha='center');text(fig,.86,.10,'Relative magnitude',8,MUTED,ha='center')
    d.save(fig,'observability_visual')

def transfer():
    forward=d.load(d.ROOT/'reports/compositional-neural-mechanics-v16/oracle-recovered-plus-stable-io-comparison.json')
    adapt=d.load(d.ROOT/'reports/compositional-neural-mechanics-v16/unknown-action-reaction-paper-summary.json')
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.1),gridspec_kw={'width_ratios':[1,1,1.05]})
    fig.subplots_adjust(left=.105,right=.97,top=.83,bottom=.25,wspace=.70)
    methods=[('matched_analytic','Matched analytic'),('neo_hookean','Neo-Hookean'),('shared_laws','Shared laws (ours)'),('custom_direct_stable','Direct / 6k'),('custom_direct_long_stable','Direct / 30k')]
    for j,cohort in enumerate(['reserved_composition','reserved_geometry']):
        a=axes[j];vals=[1000*next(r for r in forward['rows'] if r['cohort']==cohort and r['method']==m)['object_mean_scores']['displacement_coordinate_rmse_m'] for m,_ in methods]
        a.axhspan(1.65,2.35,color='#e2f1ee',zorder=0)
        for i,v in enumerate(vals):
            col=TEAL if i==2 else GOLD if i>=3 else '#8d9ca8'
            a.plot([.009,v],[i,i],color=col,lw=2,alpha=.5);a.scatter(v,i,s=45,c=col,zorder=3,edgecolors='white',linewidth=.7)
            a.text(v*1.20,i,f'{v:.3f}',va='center',fontsize=9,color=col,weight='bold' if i==2 else 'normal')
        a.set_xscale('log');a.set_xlim(.009,12);a.set_ylim(4.6,-.6);a.set_yticks(range(5),[n for _,n in methods],fontsize=9)
        a.set_xticks([.01,.1,1,10],['0.01','0.1','1','10']);a.tick_params(axis='both',length=0,labelsize=9);a.spines[['left','top','right']].set_visible(False);a.spines['bottom'].set_color('#cbd5df');a.set_axisbelow(True);a.grid(axis='x',color='#e8edf1');a.set_title(['New compositions','New geometries'][j],loc='left',fontsize=12,weight='bold',pad=12);a.set_xlabel('Position error (mm; log scale)',fontsize=9)
    a=axes[2];policies=[('initial','Initial'),('fixed_stretch','+ Stretch'),('uniform_random','+ Random'),('target_task_variance','+ Adaptive')]
    vals=[next(r for r in adapt['rows'] if r['model']=='shared_laws_fitted' and r['policy']==p)['position_rmse_mm'] for p,_ in policies]
    for i,v in enumerate(vals):
        col='#8d9ca8' if i==0 else TEAL if i in [1,3] else PURPLE
        a.plot([0,v],[i,i],color=col,lw=2,alpha=.5);a.scatter(v,i,s=48,color=col,edgecolors='white',linewidth=.7,zorder=3);a.text(v+.012,i,f'{v:.3f}',va='center',fontsize=9,color=col)
    a.axhspan(.65,1.35,color='#e2f1ee',zorder=0);a.set(xlim=(0,.43),ylim=(3.6,-.6));a.set_yticks(range(4),[n for _,n in policies],fontsize=9);a.set_xticks([0,.1,.2,.3,.4]);a.tick_params(length=0,labelsize=9);a.spines[['left','top','right']].set_visible(False);a.spines['bottom'].set_color('#cbd5df');a.set_axisbelow(True);a.grid(axis='x',color='#e8edf1');a.set_title('One extra observation',loc='left',fontsize=12,weight='bold',pad=12);a.set_xlabel('Position error (mm)',fontsize=9)
    text(fig,.5,.045,'Frozen shared laws transfer across compositions and geometries. Additional positions improve prediction; adaptive selection ties fixed stretch.',9,MUTED,ha='center')
    d.save(fig,'transfer_visual')

if __name__=='__main__':
    method();results();ambiguity();transfer();records=plate()
    (d.OUT.parent/'evidence/visual-redesign-v2.json').write_text(json.dumps(dict(scope='Presentation only; sealed data and full-query scores unchanged',selection='world00/query0/final frame, every family',geometry='Orthographic projection, shared world-space bounds within each comparison; no deformation amplification',color='Shared 0-20 mm error scale; per-triangle mean vertex error',per_family_percentage='100*(1-mean_ours_error/mean_phystwin_error), all eight queries per family',frames=records,source_sha256=d.SOURCES,script_sha256=d.digest(Path(__file__))),indent=2)+'\n')
