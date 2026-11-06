"""Publication method schematic. No experimental observations or scores depicted."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .mechanics import LinearGaussianBody

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/joint-field-identification-v01'


def main():
    torch.set_num_threads(1)
    for p in (Path.home()/'.local/share/fonts/arial').glob('*.TTF'):font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family':'Arial','font.size':14,'text.color':'black',
        'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
        'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',
        'mathtext.bf':'Arial:bold','pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(16,7.6),facecolor='white')
    ink=fig.add_axes([0,0,1,1],frameon=False);ink.set(xlim=(0,1),ylim=(0,1));ink.axis('off')
    cards={'fields':(.015,.53,.31,.43),'collapse':(.365,.53,.29,.43),'posterior':(.695,.53,.29,.43),
           'observations':(.015,.075,.31,.36),'actions':(.365,.075,.29,.36),'prediction':(.695,.075,.29,.36)}
    def card(key,title):
        x,y,w,h=cards[key];ink.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.003,rounding_size=.012',fc='white',ec='black',lw=1.1))
        ink.text(x+.012,y+h-.033,title,fontsize=17,fontweight='bold',va='top')
    for key,title in [('fields','1  Spatial physical hypotheses'),('collapse','2  Integrate unknown loads'),('posterior','3  Retain joint uncertainty'),
                      ('observations','Observed displacement tracks'),('actions','4  Look or probe?'),('prediction','5  Predict held responses')]:card(key,title)
    def arrow(a,b,style='-',curve=0):
        ink.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,lw=1.6,color='black',linestyle=style,connectionstyle=f'arc3,rad={curve}'))
    arrow((.329,.735),(.359,.735));arrow((.659,.735),(.69,.735))
    arrow((.84,.525),(.84,.44));arrow((.74,.525),(.65,.44))
    arrow((.51,.44),(.51,.525),style='--')
    ink.text(.515,.48,'new evidence',fontsize=12,ha='left',va='center')
    arrow((.328,.34),(.36,.55),curve=-.12)
    ink.text(.336,.437,'y',fontsize=15,ha='center')
    body=LinearGaussianBody('step',cells=(4,2,2));p=body.points.numpy();tri=body.surface_triangles.numpy()
    def axis3(rect):
        ax=fig.add_axes(rect,projection='3d');ax.view_init(elev=24,azim=-56)
        ax.set(xlim=(-.07,.07),ylim=(-.04,.04),zlim=(-.015,.09));ax.set_box_aspect((1.5,.8,1.05));ax.axis('off');return ax
    def mesh(ax,points=p,facecolors='#b9bec4',alpha=1.):
        pc=Poly3DCollection(points[tri],facecolors=facecolors,edgecolors=(.15,.15,.15,.16),linewidths=.28,alpha=alpha)
        ax.add_collection3d(pc)
    mat=axis3([.019,.61,.15,.255])
    centers=p[tri].mean(1);v=(centers[:,0]+.06)/.12
    cmap=colors.LinearSegmentedColormap.from_list('material',['#346dad','#65b8b0','#f4cd70'])
    mesh(mat,facecolors=cmap(v))
    loads=axis3([.171,.61,.15,.255]);mesh(loads,facecolors='#d7d9dc',alpha=.72)
    def ellipsoid(ax,center,radii,color,alpha=.3):
        u=np.linspace(0,2*np.pi,35);v=np.linspace(0,np.pi,20)
        x=center[0]+radii[0]*np.outer(np.cos(u),np.sin(v));y=center[1]+radii[1]*np.outer(np.sin(u),np.sin(v));z=center[2]+radii[2]*np.outer(np.ones_like(u),np.cos(v))
        ax.plot_surface(x,y,z,color=color,alpha=alpha,linewidth=0,shade=True)
    ellipsoid(loads,[-.027,0,.065],[.025,.037,.009],'#e88836')
    ellipsoid(loads,[.018,0,-.003],[.026,.038,.008],'#397cb9')
    for x in [-.04,-.027,-.014]:loads.quiver(x,0,.089,0,0,-.021,color='#c76a24',linewidth=2.1,arrow_length_ratio=.3)
    ink.text(.095,.592,'Material field',ha='center',fontsize=14,fontweight='bold')
    ink.text(.246,.592,'Load + support fields',ha='center',fontsize=14,fontweight='bold')
    ink.text(.17,.557,'Material contrast; unknown load / support parameters',ha='center',fontsize=11.5)
    # The three coefficient curves are illustrations of the prior, not fitted histories.
    ax=fig.add_axes([.388,.775,.244,.10]);t=np.linspace(0,1,100)
    for i in range(3):ax.plot(t,np.sin(2*np.pi*t+i*.6)*(.5+i*.2),color=str(.15+i*.2),lw=1.4)
    ax.set(xlim=(0,1),ylim=(-1.2,1.2));ax.axis('off')
    ink.text(.51,.87,'Possible force histories',ha='center',fontsize=13)
    ink.text(.51,.747,'y = H(θ, β) F + ε',ha='center',fontsize=19)
    ink.text(.51,.697,'F ~ N(0, K)',ha='center',fontsize=18)
    ink.text(.51,.638,r'$C = H K H^{T} + R$',ha='center',fontsize=18)
    ink.text(.51,.582,'Exact conditional Gaussian integral',ha='center',fontsize=13,fontweight='bold')
    # Schematic correlated posterior. It is not a plot of fitted parameters.
    ax=fig.add_axes([.744,.657,.195,.205]);x=np.linspace(-2.4,2.4,90);X,Y=np.meshgrid(x,x)
    Z=np.exp(-((X-.2)**2/.5+(Y-.8*X+.1)**2/.10)/2)+.35*np.exp(-((X+1.1)**2/.18+(Y-.8*X-.25)**2/.12)/2)
    ax.contour(X,Y,Z,levels=5,colors='black',linewidths=1)
    ax.set(xticks=[],yticks=[],xlabel='Material stiffness',ylabel='Load amplitude')
    for s in ['top','right']:ax.spines[s].set_visible(False)
    ink.text(.84,.578,'Preserve alternative coupled explanations',ha='center',fontsize=13)
    # Observed camera tracks, explicitly schematic and sparse.
    ax=fig.add_axes([.045,.17,.25,.18]);x=np.linspace(0,1,8)
    for i in range(3):
        y=.3*i+.08*np.sin(2*np.pi*x+i)
        ax.plot(x,y,color='.7',lw=1);ax.scatter(x[[0,3,7]],y[[0,3,7]],s=27,color='black',zorder=4)
    ax.set(xticks=[0,.43,1],xticklabels=['t1','t4','t8'],yticks=[])
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ink.text(.17,.12,'Known cameras, geometry and correspondences',ha='center',fontsize=11.5)
    # Camera and mechanical probe icons: acquisition inputs, not results.
    ax=fig.add_axes([.383,.19,.253,.13]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.add_patch(Rectangle((.07,.25),.22,.42,fill=False,ec='black',lw=1.7));ax.add_patch(Circle((.18,.46),.07,fill=False,ec='black',lw=1.5))
    ax.plot([.29,.43,.43,.29],[.34,.21,.73,.6],color='black',lw=1.5)
    ax.add_patch(Rectangle((.65,.08),.25,.25,fc='#9ab9c8',ec='black',lw=1))
    ax.add_patch(FancyArrowPatch((.775,.94),(.775,.35),arrowstyle='-|>',mutation_scale=18,color='black',lw=2))
    ax.text(.24,-.09,'Camera track',ha='center',fontsize=13);ax.text(.775,-.09,'Known-force probe',ha='center',fontsize=13)
    ink.text(.51,.345,'Choose the smallest expected prediction risk',ha='center',fontsize=12)
    ink.text(.51,.12,'Same target, same scalar measurement budget',ha='center',fontsize=11.5)
    # Example mesh under an illustrative, smooth deformation; not a benchmark plot.
    ax=axis3([.716,.14,.25,.19]);ghost=p.copy();ghost[:,1]+=.12*p[:,2];ghost[:,2]+=.18*(p[:,0]+.06)**2/.12
    mesh(ax,points=p,facecolors='#d0d1d4',alpha=.2);mesh(ax,points=ghost,facecolors='#7caeaf',alpha=.9)
    ink.text(.84,.345,'Hidden deformation + changed loading',ha='center',fontsize=13)
    ink.text(.84,.12,'Report prediction error and interval coverage',ha='center',fontsize=11.5)
    ink.text(.5,.028,'Method schematic. Mesh geometry is real; fields, curves and deformation are illustrative, not experimental results.',ha='center',fontsize=12)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'method.png',dpi=220,bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/'method.pdf',bbox_inches='tight',pad_inches=.025)
    (OUT/'method-provenance.json').write_text(json.dumps(dict(kind='method schematic',source=__file__,
        geometry='Reference stepped tetrahedral body',experimental_data=False,
        appearance='Arial; all text black; color denotes physical fields only',
        disclosures='Prior/posterior curves, footprint ellipsoids and output deformation are illustrative.'),indent=2))


if __name__=='__main__':main()
