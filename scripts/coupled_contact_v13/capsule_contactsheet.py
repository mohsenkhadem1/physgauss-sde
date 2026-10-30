"""Fixed development-only actual capsule trajectories; not prediction evidence."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.capsule_development import build
from scripts.coupled_contact_v13.capsule_visualization import draw_capsule,capsule_bounds
from scripts.coupled_contact_v13.render import boundary
from scripts.coupled_contact_v13.campaign import write
TIMES=[0.,.08,.15,.3,.6]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);args=parser.parse_args()
    if '/' in args.version or '..' in args.version:raise ValueError('Simple diagnostic version required')
    rows=[]
    for E,mu in [(6000.,.35),(3000.,.1)]:
        status=ROOT/'runs/coupled-contact-v13/capsule-diagnostic'/args.version/f'{E}-{mu}-8-0.0015/status.json'
        r=json.loads(status.read_text())
        if not r['success']:raise RuntimeError('Declared development clip unavailable')
        path=ROOT/r['path']/'trajectory.npz'
        with np.load(path) as a:arrays={k:a[k].copy() for k in ['frames','times','tets']}
        if not np.isfinite(arrays['frames']).all():raise ValueError('Nonfinite geometry')
        indices=[int(np.argmin(abs(arrays['times']-t))) for t in TIMES]
        if not np.allclose(arrays['times'][indices],TIMES,atol=1e-10,rtol=0):raise ValueError('Missing fixed frame')
        rows.append((E,mu,path,arrays,indices))
    rail=build('edge_drop').colliders[-1];lower,upper=capsule_bounds(rail)
    points=np.concatenate([r[3]['frames'].reshape(-1,3) for r in rows]+[lower[None],upper[None],np.array([[0.,0.,0.]])])
    lo=points.min(0)-.008;hi=points.max(0)+.008;center=(lo+hi)/2;span=max(hi-lo);lo=center-span/2;hi=center+span/2
    fig=plt.figure(figsize=(20,11));panels=[]
    for row,(E,mu,path,a,indices) in enumerate(rows):
        tri=boundary(a['tets'])
        for col,(t,index) in enumerate(zip(TIMES,indices)):
            ax=fig.add_subplot(2,5,row*5+col+1,projection='3d')
            ax.add_collection3d(Poly3DCollection(a['frames'][index][tri],facecolor='#46a5ab',edgecolor='#276970',linewidth=.15,alpha=.9))
            draw_capsule(ax,rail,color='#d48956',alpha=.7,linewidth=0)
            x,y=np.meshgrid([lo[0],hi[0]],[lo[1],hi[1]]);ax.plot_surface(x,y,np.zeros_like(x),color='#aeb2ba',alpha=.15)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=f'E={E/1000:g} kPa, μ={mu:g} | t={t:.2f}s',xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
            ax.set_box_aspect((1,1,1));ax.view_init(22,-65)
            ax.tick_params(labelsize=7,pad=0);ax.title.set_fontsize(10)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.label.set_fontsize(8)
        panels.append(dict(young=E,friction=mu,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frame_indices=indices,times_s=TIMES))
    fig.suptitle('Development diagnostic: 8-cell actual FEM trajectories on rounded capsule rail\nFixed times, common physical axes, no deformation exaggeration; not qualified prediction results',fontsize=15)
    fig.subplots_adjust(left=.025,right=.975,bottom=.09,top=.84,wspace=.05,hspace=.32)
    destination=ROOT/'reports/coupled-surface-v13';destination.mkdir(parents=True,exist_ok=True)
    image=destination/'capsule-development-contactsheet.png';fig.savefig(image,dpi=140);plt.close(fig)
    write(destination/'capsule-development-contactsheet.json',dict(scope=__doc__,sources=panels,bounds_m=[lo.tolist(),hi.tolist()],image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),drawing_sha256=hashlib.sha256((Path(__file__).with_name('capsule_visualization.py')).read_bytes()).hexdigest()))
    print(image)
if __name__=='__main__':main()
