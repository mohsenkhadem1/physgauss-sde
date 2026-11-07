"""Render saved numerical states at their true spatial scale, never generated images."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .bowl_physics import command

for font in Path('/home/bayespc2/.local/share/fonts/arial').glob('*.TTF'):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':'Arial','text.color':'black','font.size':11})

def scene(ax,r,k,title='',azimuth=-58,errors=None,zmax=.12):
    x=r['frames'][k];f=r['faces']
    colors='#dc9951' if errors is None else plt.get_cmap('YlOrRd')(np.clip(errors[f].mean(1)/.006,0,1))
    ax.add_collection3d(Poly3DCollection(x[f],facecolor=colors,edgecolor='#795c40',linewidth=.06,alpha=1))
    phi=np.linspace(0,2*np.pi,45);theta=np.linspace(np.pi/2,np.pi,19)
    t,p=np.meshgrid(theta,phi,indexing='ij')
    X=.09*np.sin(t)*np.cos(p);Y=.09*np.sin(t)*np.sin(p);Z=.09*np.cos(t)
    state=command(float(r['times'][k]),str(r['interaction']));rotation=state.get('bowl',{}).get('rotation',np.eye(3))
    xyz=np.stack((X,Y,Z),axis=-1)@rotation.T;X,Y,Z=xyz[...,0],xyz[...,1],xyz[...,2]+.09
    ax.plot_wireframe(X,Y,Z,color='#749bb1',linewidth=.35,alpha=.35,rstride=2,cstride=2)
    ax.plot_surface(X,Y,Z,color='#b2d1e1',alpha=.08,linewidth=0,shade=False)
    if 'centers' in r:
        c=r['centers'][k,1]
        if np.linalg.norm(c[:2])<.13 and c[2]-.018<=zmax:
            a,b=np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,16))
            ax.plot_surface(c[0]+.018*np.sin(b)*np.cos(a),c[1]+.018*np.sin(b)*np.sin(a),c[2]+.018*np.cos(b),color='#777777',alpha=.7,linewidth=0)
    ax.set(xlim=(-.09,.09),ylim=(-.09,.09),zlim=(-.005,zmax))
    ax.set_box_aspect((1,1,(zmax+.005)/.18),zoom=1.35);ax.view_init(24,azimuth);ax.set_axis_off()
    ax.set_title(title,pad=-4,color='black')

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',default='output/gap-first-v01/bowl/development.png');args=p.parse_args()
    r=dict(np.load(args.input));indices=np.linspace(0,len(r['frames'])-1,4).round().astype(int)
    fig=plt.figure(figsize=(12,3.4))
    for j,k in enumerate(indices):scene(fig.add_subplot(1,4,j+1,projection='3d'),r,k,f"{r['times'][k]:.2f} s",zmax=.18)
    fig.suptitle('Development: soft curved object falls into a rigid bowl',y=.98)
    fig.text(.5,.025,'Actual nonlinear FEM geometry; true scale. Open hemispherical wall; penalty contact.',ha='center',fontsize=10)
    fig.subplots_adjust(left=0,right=1,top=.87,bottom=.08,wspace=-.05)
    fig.savefig(args.output,dpi=160);plt.close(fig)


def compare(reference,initial,corrected,output_prefix,labels=None,video=True):
    """Fixed case, actual saved states; all panels share camera and spatial scale."""
    paths=[Path(p) for p in (reference,initial,corrected)]
    data=[dict(np.load(p)) for p in paths]
    labels=labels or ['Reference','Before calibration','After calibration']
    mappings=[]
    for r in data[1:]:
        distance=np.linalg.norm(r['rest'][:,None]-data[0]['rest'][None],axis=2)
        ids=distance.argmin(1)
        if distance.min(1).max()>1e-10:raise ValueError('Comparison meshes lack exact shared material nodes')
        mappings.append(ids)
    errors=[None]+[np.linalg.norm(r['frames']-data[0]['frames'][:,ids],axis=-1) for r,ids in zip(data[1:],mappings)]
    indices=[np.argmin(abs(data[0]['times']-t)) for t in (.12,.94,1.2)]
    fig=plt.figure(figsize=(10,7))
    for row,k in enumerate(indices):
        for col,r in enumerate(data):
            title=(labels[col]+'\n' if row==0 else '')+f"{r['times'][k]:.2f} s"
            scene(fig.add_subplot(3,3,3*row+col+1,projection='3d'),r,k,title,errors=None if col==0 else errors[col][k])
    fig.subplots_adjust(left=0,right=1,bottom=.10,top=.94,wspace=-.08,hspace=-.1)
    coloraxis=fig.add_axes([.37,.055,.26,.014])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,6),cmap='YlOrRd'),cax=coloraxis,orientation='horizontal',ticks=[0,2,4,6])
    bar.ax.tick_params(labelsize=10.5,pad=2)
    fig.text(.5,.012,'Position error (mm). Actual nonlinear simulation; identical camera and spatial scale.',ha='center',fontsize=11)
    prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(prefix)+'.png',dpi=200);plt.close(fig)
    if video:
        with tempfile.TemporaryDirectory(prefix='bowl-frames-') as directory:
            for k in range(len(data[0]['frames'])):
                fig=plt.figure(figsize=(11,4),dpi=120)
                for col,r in enumerate(data):
                    scene(fig.add_subplot(1,3,col+1,projection='3d'),r,k,labels[col],azimuth=-65+35*k/(len(r['frames'])-1),errors=None if col==0 else errors[col][k],zmax=.18)
                fig.suptitle(f"Soft object in a rigid bowl • physical time {data[0]['times'][k]:.2f} s",fontsize=15,y=.97)
                fig.text(.5,.035,'Gravity drop → transverse push. Actual saved states; slow playback; no deformation amplification.',ha='center',fontsize=11)
                coloraxis=fig.add_axes([.42,.11,.16,.018])
                bar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,6),cmap='YlOrRd'),cax=coloraxis,orientation='horizontal',ticks=[0,3,6])
                bar.ax.tick_params(labelsize=9,pad=1)
                fig.text(.40,.112,'Position error (mm)',ha='right',fontsize=10)
                fig.subplots_adjust(left=0,right=1,top=.85,bottom=.19,wspace=-.06)
                fig.savefig(Path(directory)/f'{k:04d}.png');plt.close(fig)
            subprocess.run(['runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','15','-i',str(Path(directory)/'%04d.png'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(prefix)+'.mp4'],check=True)
    manifest=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        real_geometry_scale=True,comparison_camera_identical=True,error_color_max_mm=6,physical_duration_s=float(data[0]['times'][-1]),playback_fps=15,
        note='Reference is simulated; no photorealistic rendering claim. Positions are actual saved nonlinear trajectories, not interpolated animation.',
        source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(str(prefix)+'-manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
