"""Predeclared median/worst held-case selection and error-colored paper panels."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,ROOT,OUT
from scripts.position_ood.runtime import CONFIG
from scripts.position_ood.score import run as score
from scripts.position_ood.seal import verify
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write

def faces():
 shape=(9,3,3);rows=[]
 for axis in range(3):
  other=[j for j in range(3) if j!=axis]
  for side in [0,shape[axis]-1]:
   for a in range(shape[other[0]]-1):
    for b in range(shape[other[1]]-1):
     row=[]
     for da,db in [(0,0),(1,0),(1,1),(0,1)]:
      ix=[0,0,0];ix[axis]=side;ix[other[0]]=a+da;ix[other[1]]=b+db;row.append(np.ravel_multi_index(ix,shape))
     rows.append(row)
 return np.array(rows)

def main(video=False):
 seal=verify();config=json.loads(CONFIG.read_text());ranked=[]
 for split,indices in config['expensive_baseline_indices'].items():
  report=score(split);lookup={r['object']:r for r in report['rows']}
  for i in indices:
   row=lookup[i];methods=row['methods'];required=['extrapolated-n3-p3','phystwin-'+seal['settings']['phystwin']['tag'],'physgaussian-'+seal['settings']['mpm']['tag']]
   if any(m not in methods for m in required):raise RuntimeError('Complete predeclared physical subset required for qualitative selection')
   errors=np.asarray(methods['extrapolated-n3-p3']['coordinate_rmse_mm']);ranked.append(dict(split=split,object=i,held_mean_mm=float(errors[3:].mean()),action=int(np.argmax(errors[3:])+3)))
 ranked.sort(key=lambda r:(r['held_mean_mm'],r['split'],r['object']));selected=[dict(rank='Median-ranked',**ranked[len(ranked)//2]),dict(rank='Largest-error',**ranked[-1])];write(OUT/'qualitative-selection.json',dict(rule=config['qualitative_selection'],ranked=ranked,selected=selected))
 fig=plt.figure(figsize=(12,5.5));quad=faces();norm=Normalize(0,2);cmap=plt.get_cmap('magma');names=['Reference','Spatial inverse','PhysTwin core adaptation','PhysGaussian MPM adaptation'];rest=points()
 for ri,row in enumerate(selected):
  split,i,action=row['split'],row['object'],row['action'];truth=np.load(BASE/f'fields/{split}-{i}.npz')['field'][action];paths=[None,BASE/f'inverse/{split}-{i}/extrapolated-n3-p3.npz',BASE/f'phystwin/{split}-{i}/{seal["settings"]["phystwin"]["tag"]}/predictions.npz',BASE/f'{seal["settings"]["mpm"]["tag"]}/{split}-{i}/predictions.npz']
  for col,path in enumerate(paths):
   field=truth if path is None else np.load(path)['predicted'][action];error=1000*np.linalg.norm(field-truth,axis=1);xyz=1000*(rest+field);ax=fig.add_subplot(2,4,ri*4+col+1,projection='3d');colors=np.tile([.6,.7,.72,1],(len(quad),1)) if path is None else cmap(norm(error[quad].mean(1)));ax.add_collection3d(Poly3DCollection(xyz[quad],facecolors=colors,edgecolors=(0,0,0,.4),linewidths=.4));ax.set_xlim(-15,255);ax.set_ylim(-75,75);ax.set_zlim(-90,60);ax.set_box_aspect((2.7,1.5,1.5));ax.view_init(22,-65);ax.set_axis_off()
   if ri==0:ax.set_title(names[col],fontsize=10,pad=0)
   label=f'{row["rank"]}: {split.replace("test_","")}, object {i}' if col==0 else f'Coordinate RMSE: {1000*np.sqrt(np.mean((field-truth)**2)):.3f} mm'
   ax.text2D(.02,.01,label,transform=ax.transAxes,fontsize=8)
  if video:
   subprocess.run([str(ROOT/'runs/phystwin-env/bin/python'),str(ROOT/'scripts/position_ood/render.py'),'--split',split,'--object',str(i),'--phystwin-tag',seal['settings']['phystwin']['tag'],'--mpm-tag',seal['settings']['mpm']['tag']],check=True)
 fig.subplots_adjust(left=.01,right=.94,bottom=.04,top=.94,wspace=-.02,hspace=.08);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.955,.22,.012,.55]),extend='max');cb.set_label('Euclidean point error (mm)',fontsize=9);dest=ROOT/'papers'/BASE.name/'figures';fig.savefig(dest/'qualitative.pdf',bbox_inches='tight');fig.savefig(dest/'qualitative.png',dpi=170,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args();main(a.video)
