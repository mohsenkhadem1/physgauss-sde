"""First-page teaser and explicit fit/forecast graph from audited results.

The teaser selects curved_push and container_drop, world00/query0/final frame.
Zooms use the same projected world-space rectangle for all methods in a row.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import design_editorial_figures as e
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
d=e.d

def teaser():
    fig=plt.figure(figsize=(12,3.9));can=fig.add_axes([0,0,1,1]);can.axis('off')
    methods=['reference','ours','phystwin','spring-gaus'];records=[]
    for j,m in enumerate(methods):e.text(fig,.245+j*.205,.94,'Reference' if m=='reference' else d.NAMES[m],11,d.COLORS.get(m,e.INK),'bold',ha='center')
    for i,f in enumerate(['curved_push','container_drop']):
        items,t=d.selected(f);y=.52-i*.40
        allx=[items[m][-1] for m in methods];lim=e.extent(allx)
        refp=e.project(items['reference'][-1].reshape(-1,3))[:,:2]
        lo,hi=refp.min(0),refp.max(0);span=hi-lo
        # Fixed relative location on reference, identical world-space window across methods.
        center=lo+span*np.array([.53,.27 if i==0 else .43]);half=span*np.array([.12,.15]);crop=(center-half,center+half)
        title=['Curved\ncontact','Drop +\nobstacle'][i]
        e.text(fig,.015,y+.21,title,11,e.INK,'bold',linespacing=1.25)
        e.text(fig,.015,y+.09,f'{t[-1]:.3f} s\nheld interaction',8,e.MUTED,linespacing=1.5)
        errs={}
        for j,m in enumerate(methods):
            left=.15+j*.205;xx=items[m][-1];err=np.linalg.norm(xx-items['reference'][-1],axis=-1)*1000
            a=e.obj(fig,[left,y+.03,.185,.31],xx,lim,error=None if m=='reference' else err,base=d.BLUE)
            a.add_patch(Rectangle(crop[0],*(crop[1]-crop[0]),fill=False,edgecolor='#d55248',lw=.8,zorder=20))
            zoom=e.obj(fig,[left+.117,y-.015,.080,.15],xx,crop,error=None if m=='reference' else err,base=d.BLUE)
            zoom.set_aspect('equal',adjustable='box');zoom.axis('on');zoom.set_xticks([]);zoom.set_yticks([])
            for s in zoom.spines.values():s.set_visible(True);s.set_edgecolor('#d55248');s.set_linewidth(.85)
            # Panel insets are magnified crops, never independently rescaled objects.
            if m!='reference':
                rms=float(np.sqrt(np.mean(err**2)));errs[m]=rms
                e.text(fig,left+.02,y-.005,f'{rms:.2f} mm',9,d.COLORS[m],'bold' if m=='ours' else 'normal')
        records.append(dict(family=f,world=f+'-world00',query=0,time_s=float(t[-1]),zoom_world_projected_bounds=[c.tolist() for c in crop],frame_rms_mm=errs))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=d.NORM,cmap=d.CMAP),cax=fig.add_axes([.60,.035,.36,.014]),orientation='horizontal',extend='max');cb.ax.tick_params(labelsize=7,length=2,pad=1)
    e.text(fig,.015,.02,'Predict changed interactions, not just observed motion.',10,e.TEAL,'bold')
    e.text(fig,.60,-.035,'Position error (mm); shared scale, >20 mm saturates',8,e.MUTED)
    can.set(xlim=(0,1),ylim=(0,1));d.save(fig,'first_page_teaser');return records

def graph():
    fig=plt.figure(figsize=(12,4.2));can=fig.add_axes([0,0,1,1]);can.axis('off')
    def box(x,y,w,h,title,detail,color,fs=10):
        can.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.014',facecolor=color+'0D',edgecolor=color,lw=1))
        e.text(fig,x+w/2,y+h-.045,title,fs,color,'bold',ha='center');e.text(fig,x+w/2,y+.045,detail,9,e.INK,ha='center',va='center')
    e.text(fig,.015,.95,'A  SCENE ADAPTATION',11,d.BLUE,'bold');e.text(fig,.32,.95,'B  SHARED MECHANICS',11,e.TEAL,'bold');e.text(fig,.81,.95,'C  NEW INTERACTION',11,e.GOLD,'bold')
    x=e.block();lim=e.extent([x])
    for i in range(3):
        xx=x.copy();xx[:,2]+=.05*i*np.sin(np.pi*d.Q[:,0]);e.obj(fig,[.015+i*.04,.66-i*.055,.13,.19],xx[None],lim,base=['#bfd6e8','#81afcf',d.BLUE][i])
    e.text(fig,.015,.52,'Observed positions',10,d.BLUE,'bold');e.text(fig,.015,.46,r'$\mathcal{D}=\{X_i,x_i(t)\}$',10)
    box(.015,.22,.205,.15,'FIT SCENE DESCRIPTORS',r'$z_m(X),\ z_b$',e.GOLD,9)
    e.arrow(can,(.10,.44),(.10,.38),d.BLUE)
    e.text(fig,.015,.13,'Geometry, density and\ncorrespondences supplied',8,e.MUTED,linespacing=1.3)
    # Two model branches: learned local response and explicit scene interactions.
    box(.31,.61,.20,.20,'BULK ENERGY','',e.TEAL)
    e.text(fig,.445,.675,r'$W_\theta(F;z_m)$',9,ha='center')
    e.text(fig,.445,.635,r'$P=\partial W/\partial F$',8,ha='center')
    for xx,ys in [(.33,[.65,.70]),(.355,[.64,.675,.71]),(.38,[.65,.70])]:
        can.scatter([xx]*len(ys),ys,s=10,color=e.TEAL,zorder=5)
    for a,ya,b,yb in [(.33,[.65,.70],.355,[.64,.675,.71]),(.355,[.64,.675,.71],.38,[.65,.70])]:
        for aa in ya:
            for bb in yb:can.plot([a,b],[aa,bb],color='#8fbcb7',lw=.6)
    box(.31,.33,.20,.20,'ATTACHMENT ENERGY','',e.TEAL,9)
    e.text(fig,.45,.385,r'$\psi_\phi(\delta;z_b)$',9,ha='center')
    can.add_patch(Rectangle((.325,.37),.013,.065,facecolor=e.GOLD,edgecolor='none'))
    can.add_patch(Rectangle((.385,.37),.013,.065,facecolor=e.TEAL,edgecolor='none'))
    can.plot(np.linspace(.34,.385,9),.402+np.array([0,1,-1,1,-1,1,-1,1,0])*.012,color=e.MUTED,lw=1)
    for yy in [.565,.285]:e.text(fig,.315,yy,'FROZEN SHARED WEIGHTS',7,e.TEAL,'bold')
    e.arrow(can,(.225,.32),(.30,.70),e.GOLD,rad=-.15)
    e.arrow(can,(.225,.29),(.30,.43),e.GOLD,rad=.1)
    box(.57,.43,.18,.22,'Q2 MECHANICS',r'$M\ddot u=-\nabla_u\Pi$',e.INK)
    e.arrow(can,(.515,.70),(.565,.60),e.TEAL);e.arrow(can,(.515,.43),(.565,.49),e.TEAL)
    box(.56,.72,.19,.15,'QUERY / ENVIRONMENT','Tool poses / gravity\nContact / initial state',e.GOLD,8)
    e.arrow(can,(.655,.695),(.655,.66),e.GOLD)
    e.text(fig,.57,.34,'Compose bulk, attachments,\nnormal contact and body loads',8,e.MUTED,linespacing=1.4)
    # Forecast branches. No unknown force history is supplied.
    e.arrow(can,(.755,.56),(.81,.71),e.TEAL)
    xx=x.copy();xx[:,2]+=.22*d.Q[:,0]**2;e.obj(fig,[.815,.61,.17,.23],xx[None],e.extent([xx]),base=e.TEAL)
    e.text(fig,.90,.56,'Physical state',10,e.TEAL,'bold',ha='center');e.text(fig,.90,.50,r'$x(X,t),\ F,\ \mathrm{reactions}$',9,ha='center')
    e.arrow(can,(.90,.46),(.90,.37),e.GOLD)
    box(.81,.19,.18,.16,'GAUSSIAN TRANSPORT',r'$\mu_t,\ \Sigma_t\ \to\ \mathrm{RGB}$',e.GOLD,8.5)
    e.text(fig,.81,.135,'Appearance held fixed',8,e.MUTED)
    # Operational fit feedback, rather than claiming a differentiable held fitting algorithm.
    can.plot([.81,.785,.785,.25,.25],[.54,.54,.17,.17,.27],linestyle='--',color=e.GOLD,lw=1.1)
    can.add_patch(FancyArrowPatch((.25,.27),(.22,.27),arrowstyle='-|>',mutation_scale=12,linestyle='--',color=e.GOLD,lw=1.1))
    # Mask label over the long bottom return path.
    e.text(fig,.32,.08,'CALIBRATION ONLY: position residual → update descriptors',9,e.GOLD,'bold',bbox=dict(facecolor='white',edgecolor='none',pad=2))
    e.text(fig,.32,.025,'FORECAST: freeze fitted descriptors; change the interaction and roll forward.',9,e.INK)
    can.set(xlim=(0,1),ylim=(0,1));d.save(fig,'method_graph')

if __name__=='__main__':
    records=teaser();graph()
    (d.OUT.parent/'evidence/teaser-method-design.json').write_text(json.dumps(dict(scope='Presentation only; illustrative choices disclosed, full-cohort results retained elsewhere',selection='curved_push and container_drop, world00/query0/final frame; chosen to illustrate localized contact and changed drop response',zoom='Identical projected world-space crop across methods, chosen from reference geometry only',rows=records,source_sha256=d.SOURCES,script_sha256=d.digest(Path(__file__)),design_references=['https://arxiv.org/abs/2311.12198','https://arxiv.org/abs/2503.17973','https://arxiv.org/abs/2505.16971']),indent=2)+'\n')
