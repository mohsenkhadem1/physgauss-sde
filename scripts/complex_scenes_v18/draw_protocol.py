"""Paper-ready vector diagram of the common-input changed-interaction benchmark."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scripts.complex_scenes_v18.run_development import ROOT


def main():
    out=ROOT/'output/complex-scenes-v18';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,6.5));ax.set(xlim=(0,13),ylim=(0,6.5));ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',facecolor=color,edgecolor='#64748b',linewidth=1))
        ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontsize=11,weight='bold',color='#0f172a')
        ax.text(x+w/2,y+h/2-.13,body,ha='center',va='center',fontsize=10,color='#334155',linespacing=1.45)
    def arrow(a,b,color='#475569',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,linewidth=1.4,linestyle=style))
    ax.text(.2,6.2,'Infer once. Predict a new interaction. Evaluate physics and images separately.',fontsize=16,weight='bold',color='#0f172a')
    box(.2,3.55,2.4,1.8,'Observed calibration','Visible 3D point tracks\nShort motion prefix\nKnown rest geometry','#e0f2fe')
    box(3.1,3.55,3,1.8,'Fit and seal material','Ours: shared neural material law\n+ spatial material descriptor\nBaselines: springs / material latent','#dbeafe')
    box(6.6,3.55,2.5,1.8,'Predict motion','Same object material\nNew prescribed interaction\nNo refitting to query targets','#dcfce7')
    box(9.6,3.55,3.1,1.8,'Evaluate physical response','Motion and deformation\nContact, impulse, sampled folds\nFailures and coverage','#fef3c7')
    arrow((2.64,4.45),(3.06,4.45));arrow((6.14,4.45),(6.56,4.45));arrow((9.14,4.45),(9.56,4.45))
    box(.2,.9,5.9,1.9,'Public scene and action inputs','Fixture positions / release • curved obstacles • gravity and drops\nMoving rigid strikers • two deformable objects\nKnown geometry and boundary parameters\nSpatial material variation remains hidden','#f1f5f9')
    arrow((2,2.84),(2,3.51));arrow((5.2,2.84),(5.2,3.51));ax.plot([6.14,6.35,6.35],[2.1,2.1,3.15],color='#475569',linewidth=1.4);arrow((6.35,3.15),(7.,3.51))
    box(6.6,.9,2.5,1.9,'Shared rendering','Frozen Gaussian appearance\nSame motion transport\nFixed withheld cameras','#ede9fe')
    box(9.6,.9,3.1,1.9,'Evaluate image prediction','PSNR / SSIM / LPIPS\nSilhouette and depth\nObject-only rendering','#ede9fe')
    arrow((7.85,3.51),(7.85,2.84));arrow((9.14,1.85),(9.56,1.85))
    ax.text(.2,.3,'Reference-only hidden materials never enter fitting or prediction. Average queries within each object before comparing methods.',fontsize=10,color='#475569')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    for suffix in ('svg','pdf','png'):fig.savefig(out/f'benchmark-protocol.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
