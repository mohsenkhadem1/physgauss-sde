"""Presentation-only figures: schematic pipeline and sealed held trajectories.

PYTHONPATH=. .venv/bin/python papers/cvpr2027-compositional-mechanics/tools/design_visual_paper.py
No optimization, new simulations, outcome selection or changed scoring.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scripts.complex_scenes_v18.pair_contact import material_surface

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / 'figures'
REPORT = ROOT / 'reports/complex-scenes-v18/held-physical-results.json'
INK, MUTED = '#183044', '#637585'
TEAL, BLUE, GOLD, PURPLE = '#008c88', '#397bc1', '#df8b32', '#8a68ad'
COLORS = {'ours': TEAL, 'analytic-control': '#9aa6b1', 'phystwin': GOLD, 'spring-gaus': PURPLE}
NAMES = {'ours': 'Ours', 'analytic-control': 'Analytic control', 'phystwin': 'PhysTwin', 'spring-gaus': 'Spring-Gaus'}
FAMILIES = ['clamp_release', 'curved_push', 'container_drop', 'rigid_impact', 'deformable_pair', 'heterogeneous']
LABELS = ['Clamps / release', 'Curved pushing', 'Container drop', 'Rigid impact', 'Two soft bodies', 'Heterogeneous']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                     'axes.labelcolor': INK, 'text.color': INK, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
Q, FACES = material_surface(9)
SOURCES = {}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    SOURCES[str(p.relative_to(ROOT))] = digest(p)
    return json.loads(p.read_text())
DATA = load(REPORT)['rows']
def trajectory(p):
    seal=json.loads((p.parent/'complete.json').read_text())
    assert seal['completed'] and seal['trajectory_sha256']==digest(p)
    SOURCES[str(p.relative_to(ROOT))] = digest(p)
    with np.load(p) as a:
        x = a['surface_positions_m'].copy()
        assert not np.iscomplexobj(x) or not np.any(x.imag)
        return x.real, a['times_s'].copy()
def save(fig, name):
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight', pad_inches=.025)
    fig.savefig(OUT / (name+'.png'), dpi=190, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
def camera(ax, bounds):
    lo, hi = bounds
    ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), zlim=(lo[2],hi[2]))
    ax.set_box_aspect(hi-lo); ax.view_init(23, -64); ax.set_axis_off()
    ax.set_proj_type('ortho'); ax.patch.set_alpha(0)
def bounds(x, pad=.08):
    lo=x.reshape(-1,3).min(0); hi=x.reshape(-1,3).max(0)
    margin=max(hi-lo)*pad
    return lo-margin,hi+margin
def surface(ax,x,color=TEAL,error=None):
    polys=x[FACES]
    if error is None:
        normals=np.cross(polys[:,1]-polys[:,0],polys[:,2]-polys[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-14)
        light=np.abs(normals@np.array([.3,-.4,.86]))
        colors=np.clip(np.array(to_rgb(color))[None,:]*(.65+.35*light[:,None]),0,1)
    else: colors=CMAP(NORM(error[FACES].mean(1)))
    ax.add_collection3d(Poly3DCollection(polys,facecolors=colors,edgecolors=(.12,.22,.29,.12),linewidths=.16,rasterized=True))
def schematic(ax, mode):
    x=Q*np.array([1.8,.75,.75]);x[:,1]-=.375;x[:,2]-=.375
    x[:,2]+=.20*np.sin(np.pi*Q[:,0]); b=bounds(x,.15)
    if mode=='gaussian':
        ax.scatter(*x[::3].T,s=13,c=Q[::3,0],cmap='viridis',alpha=.45,edgecolors='none',rasterized=True)
    else:
        surface(ax,x,TEAL if mode!='observe' else BLUE, error=Q[:,0]*12 if mode=='material' else None)
        if mode=='observe':
            ax.scatter(*x[::24].T,s=14,color=GOLD,depthshade=False)
        if mode=='solve':
            ax.quiver(.9,0,.95,0,0,-.45,color=GOLD,arrow_length_ratio=.35,linewidth=2)
            ax.plot([0,0],[-.5,.5],[-.4,-.4],lw=5,color=MUTED)
    camera(ax,b)
def pipeline():
    fig=plt.figure(figsize=(11.8,3.75)); ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    xs=[.012,.212,.412,.612,.812];w=.177
    titles=['Observe positions','Fit scene descriptors','Shared local laws','Compose & solve','Predict & render']
    subtitles=['Geometry + mass supplied','Material / attachment state','Pretrain once; freeze weights','Query poses + gravity + contact','New motion + held views']
    colors=[BLUE,PURPLE,TEAL,TEAL,GOLD]
    for i,(x,title,sub,col) in enumerate(zip(xs,titles,subtitles,colors)):
        ax.add_patch(FancyBboxPatch((x,.25),w,.69,boxstyle='round,pad=.006,rounding_size=.015',facecolor='#f5f8fb',edgecolor='#dbe3e9',lw=.9))
        ax.text(x+.012,.885,f'{i+1:02}',fontsize=10,color=col,weight='bold')
        ax.text(x+w/2,.82,title,fontsize=10,ha='center',weight='bold')
        ax.text(x+w/2,.295,sub,fontsize=7.4,ha='center',color=MUTED)
        if i<4:ax.add_patch(FancyArrowPatch((x+w+.007,.58),(xs[i+1]-.01,.58),arrowstyle='-|>',mutation_scale=13,color=MUTED,lw=1.3))
        if i!=2:
            a=fig.add_axes([x+.005,.36,w-.01,.41],projection='3d')
            schematic(a,['observe','material',None,'solve','gaussian'][i])
        else:
            a=fig.add_axes([x+.025,.39,w-.05,.33],projection='3d')
            u,v=np.meshgrid(np.linspace(-1,1,28),np.linspace(-1,1,28));z=u*u+.6*v*v+.3*u**4
            a.plot_surface(u,v,z,cmap='viridis',linewidth=0,rasterized=True);a.view_init(25,-65);a.set_axis_off()
            ax.text(x+w/2,.355,r'$W_\theta(F,z_m)\quad+\quad\psi_\phi(\delta,z_b)$',ha='center',fontsize=9)
    ax.text(.018,.145,'POSITION INPUT',color=BLUE,fontsize=8,weight='bold')
    ax.text(.185,.145,'Fit descriptors, not network weights',fontsize=8,color=MUTED)
    ax.text(.535,.145,'PHYSICS OUTPUT',color=TEAL,fontsize=8,weight='bold')
    ax.text(.69,.145,'Deformation / reactions',fontsize=8)
    ax.text(.535,.06,'VISUAL OUTPUT',color=GOLD,fontsize=8,weight='bold')
    ax.text(.69,.06,'Transported Gaussian appearance',fontsize=8)
    ax.text(.018,.055,'Schematic components; physical and image errors are measured separately.',fontsize=8,color=MUTED)
    save(fig,'method_visual')

CMAP=LinearSegmentedColormap.from_list('position_error',['#19a6a0','#9fd8c9','#f5e6a2','#e99b48','#b64135'])
NORM=Normalize(0,20,clip=True)
def selected(family):
    world=family+'-world00'; rows=[r for r in DATA if r['world_id']==world and r['query_index']==0]
    refpath=next((ROOT/'runs/complex-scenes-v18/held-campaign-01/references'/world/'0').glob('*/trajectory.npz'))
    ref,t=trajectory(refpath); items={'reference':ref}
    for m in COLORS:
        r=next(r for r in rows if r['method']==m);assert r['state']=='complete'
        x,clock=trajectory(Path(r['physical_output'])/'trajectory.npz');assert x.shape==ref.shape and np.allclose(t,clock,atol=1e-9,rtol=0);items[m]=x
    return items,t

def error_plate():
    fig=plt.figure(figsize=(11.8,12.6))
    methods=['reference','ours','analytic-control','phystwin','spring-gaus']
    for j,m in enumerate(methods):
        fig.text(.20+j*.175,.965,'Reference' if m=='reference' else NAMES[m],ha='center',fontsize=12,weight='bold',color=COLORS.get(m,INK))
    records=[]
    for i,(family,label) in enumerate(zip(FAMILIES,LABELS)):
        items,t=selected(family); y=.81-i*.135
        allx=np.concatenate([x[-1].reshape(-1,3) for x in items.values()]);b=bounds(allx,.06)
        fig.text(.012,y+.057,label.replace(' / ',' /\n'),fontsize=9,weight='bold',va='center')
        fig.text(.012,y+.018,f'$t={t[-1]:.3f}$ s',fontsize=8,color=MUTED)
        errors={}
        for j,m in enumerate(methods):
            ax=fig.add_axes([.112+j*.175,y,.175,.125],projection='3d')
            frame=items[m][-1]
            err=np.linalg.norm(frame-items['reference'][-1],axis=-1)*1000
            for k,body in enumerate(frame):surface(ax,body,color=BLUE if k==0 else PURPLE,error=None if m=='reference' else err[k])
            camera(ax,b)
            if m!='reference':
                rms=float(np.sqrt(np.mean(err**2)));errors[m]=rms
                fig.text(.20+j*.175,y-.006,f'{rms:.2f} mm',ha='center',fontsize=9,weight='bold' if m=='ours' else 'normal',color=COLORS[m])
        records.append(dict(family=family,world=family+'-world00',query=0,time_s=float(t[-1]),frame_vector_rms_mm=errors))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=fig.add_axes([.32,.06,.46,.012]),orientation='horizontal',extend='max')
    cb.set_label('Point-position error (mm) | shared color scale; values above 20 mm saturate',fontsize=9)
    fig.text(.5,.012,'First world in every family, query 0, final frame. Shared camera and bounds within each row. True deformation scale.',ha='center',fontsize=8,color=MUTED)
    save(fig,'held_error_plate')
    return records

def benchmark():
    fig=plt.figure(figsize=(11.8,6.5))
    for i,(family,label) in enumerate(zip(FAMILIES,LABELS)):
        items,t=selected(family)
        b=bounds(np.concatenate([items[m][-1].reshape(-1,3) for m in ['reference','ours','phystwin']]),.06)
        fig.text(.13+i*.153,.967,label,ha='center',fontsize=8.5,weight='bold')
        for j,m in enumerate(['reference','ours','phystwin']):
            ax=fig.add_axes([.053+i*.153,.80-j*.145,.153,.14],projection='3d')
            x=items[m][-1];err=np.linalg.norm(x-items['reference'][-1],axis=-1)*1000
            for k,body in enumerate(x):surface(ax,body,BLUE if k==0 else PURPLE,error=None if m=='reference' else err[k])
            camera(ax,b)
    for j,(name,color) in enumerate([('Reference',INK),('Ours',TEAL),('PhysTwin',GOLD)]):
        fig.text(.012,.87-j*.145,name,rotation=90,ha='center',va='center',fontsize=9,weight='bold',color=color)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=fig.add_axes([.31,.475,.39,.009]),orientation='horizontal',extend='max')
    cb.ax.tick_params(labelsize=7,pad=1);cb.set_label('Position error (mm); above 20 mm saturates',fontsize=7,labelpad=1)
    ax=fig.add_axes([.067,.105,.61,.275]); pos=np.arange(6)
    for j,m in enumerate(COLORS):
        values=[np.mean([r['metrics']['vector_rms_mm'] for r in DATA if r['family']==f and r['method']==m and r['state']=='complete']) for f in FAMILIES]
        ax.bar(pos+(j-1.5)*.19,values,width=.18,color=COLORS[m],label=NAMES[m])
    ax.set_xticks(pos,['Clamps','Curved\npush','Drop','Impact','Two\nbodies','Hetero.']);ax.set_ylabel('Motion RMS (mm)  ↓');ax.set_ylim(0,30)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e7edf1');ax.legend(ncol=4,frameon=False,fontsize=8,loc='upper left')
    ax.set_title('Changed interactions | 8 queries per family',loc='left',fontsize=9,weight='bold',pad=9)
    a=fig.add_axes([.765,.105,.22,.275]);methods=list(COLORS)
    vals=[np.mean([r['metrics']['vector_rms_mm'] for r in DATA if r['method']==m and r['state']=='complete']) for m in methods]
    a.barh(range(4),vals,color=[COLORS[m] for m in methods],height=.58);a.invert_yaxis();a.set_yticks(range(4),[NAMES[m] for m in methods],fontsize=8)
    a.set_xlim(0,14);a.set_xlabel('Motion RMS (mm)  ↓',fontsize=9);a.grid(axis='x',color='#e7edf1');a.set_axisbelow(True)
    for i,v in enumerate(vals):a.text(v+.18,i,f'{v:.2f}',va='center',fontsize=9,weight='bold' if i==0 else 'normal')
    a.set_title('All 48 queries',loc='left',fontsize=9,weight='bold',pad=9)
    fig.text(.5,.025,'Four independent worlds per family. Native method adaptations; analytic control shown. UniPhy: 1/8 supported queries completes.',ha='center',fontsize=8,color=MUTED)
    save(fig,'held_visual_results')

if __name__=='__main__':
    pipeline(); records=error_plate(); benchmark()
    p=OUT.parent/'evidence/visual-redesign.json'
    p.write_text(json.dumps(dict(scope='Presentation only; original scores and simulations unchanged',selection='world00, query0, final frame in every family, fixed before plotting',surface_color='point error averaged over each rendered triangle; common 0-20 mm clipped scale',geometry='true-scale geometry; common bounds per comparison row; orthographic camera',pipeline='Schematic illustrations, not empirical observations',frames=records,source_sha256=SOURCES,script_sha256=digest(Path(__file__))),indent=2)+'\n')
    print(p)
