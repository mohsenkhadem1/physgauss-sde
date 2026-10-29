"""Reference-only development illustration at the true deformation scale."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,ROOT,interpolate_samples
from scripts.position_ood.score import deformation

def main():
 i=3;data=np.load(BASE/f'fields/development-{i}.npz');n=int(data['mesh_n']);axes=[np.linspace(0,.24,33),np.linspace(-.04,.04,9),np.linspace(-.04,.04,9)];query=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3);shape=(33,9,9);faces=[]
 for axis in range(3):
  others=[j for j in range(3) if j!=axis]
  for side in [0,shape[axis]-1]:
   for j in range(shape[others[0]]-1):
    for k in range(shape[others[1]]-1):
     row=[]
     for a,b in [(0,0),(1,0),(1,1),(0,1)]:
      ix=[0,0,0];ix[axis]=side;ix[others[0]]=j+a;ix[others[1]]=k+b;row.append(np.ravel_multi_index(ix,shape))
     faces.append(row)
 faces=np.array(faces);stats=deformation(data['field']);actions=[0,1,2,4,6,7];names=['Nominal pose','Right tool: y +6 mm','Right tool: z +6 mm','Stretch','Twist','Bend'];fig=plt.figure(figsize=(11,5.2));norm=Normalize(0,15);cmap=plt.get_cmap('viridis')
 for panel,(action,name) in enumerate(zip(actions,names)):
  u=interpolate_samples(data['full_displacement'][action],query,n);x=query+u;X=query-query.mean(0);xc=x-x.mean(0);U,_,V=np.linalg.svd(X.T@xc);D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);nr=np.linalg.norm(xc-X@(U@D@V),axis=1)*1000
  ax=fig.add_subplot(2,3,panel+1,projection='3d');collection=Poly3DCollection((1000*x)[faces],facecolors=cmap(norm(nr[faces].mean(1))),edgecolors=(0,0,0,.14),linewidths=.15);ax.add_collection3d(collection);ax.set_xlim(-15,255);ax.set_ylim(-70,70);ax.set_zlim(-80,60);ax.set_box_aspect((2.7,1.4,1.4));ax.view_init(elev=22,azim=-65);ax.set_axis_off();ax.set_title(name,fontsize=11,pad=-2);ax.text2D(.06,.03,f'Nonrigid coordinate RMS: {stats[action]["nonrigid_rms_mm"]:.2f} mm',transform=ax.transAxes,fontsize=8)
 fig.subplots_adjust(left=.01,right=.92,bottom=.04,top=.94,wspace=-.02,hspace=.05);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.94,.2,.015,.6]));cb.set_label('Point displacement after rigid alignment (mm)',fontsize=9);cb.ax.tick_params(labelsize=8)
 dest=ROOT/'papers'/BASE.name/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'scenarios.pdf',bbox_inches='tight');fig.savefig(dest/'scenarios.png',dpi=160,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
