"""Draw the implemented architecture; no result data or scientific runs are modified."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/compositional-neural-mechanics-v16'

def main():
    fig,ax=plt.subplots(figsize=(14,8));ax.set(xlim=(0,14),ylim=(0,8));ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(.3,7.65,'Learn local laws. Compose them mechanically. Refine the scene.',fontsize=18,weight='bold',color='#173c50')
    def box(x,y,w,h,title,body,color='#eaf1f5',dashed=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#638495',lw=1.3,linestyle='--' if dashed else '-'))
        ax.text(x+.13,y+h-.28,title,fontsize=12,weight='bold',va='top',color='#173c50')
        ax.text(x+.13,y+h-.68,body,fontsize=10.3,va='top',linespacing=1.5,color='#263b45')
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#456e80',lw=1.5))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,fontsize=9,ha='center',color='#456e80')
    box(.35,4.75,3.5,2,'Known scene + commands','Geometry and law families\nGravity and density\nTool positions / orientations')
    box(.35,1.85,3.5,2.15,'Observed response','Noisy tracked 3D positions\nFit 15 continuous scene variables\nMaterial + support properties')
    box(4.65,4.75,3.9,2,'Reusable learned energies','Bulk material: energy → stress\nAttachments: relative displacement\nShared weights remain frozen','#e8f3ec')
    box(4.65,1.85,3.9,2.15,'Verified equilibrium solver','Compose energies + body loads\nSolve deformation; check residual\nImplicit derivatives enable inference')
    box(9.4,4.75,4.15,2,'Predictions for a new command','Deformation + tool forces / torques\nSame state → RGB, depth, silhouette\nKnown synthetic appearance')
    box(9.4,1.85,4.15,2.15,'Choose another observation','Local uncertainty in future response\nCompare candidate position probes\nObserve → update scene variables')
    arrow((3.9,5.7),(4.55,5.7))
    arrow((3.9,4.8),(4.55,3.8))
    arrow((3.9,2.8),(4.55,2.8))
    arrow((6.6,4.65),(6.6,4.1),'laws')
    arrow((8.65,3.6),(9.3,5.1))
    arrow((8.65,2.8),(9.3,2.8))
    arrow((11.4,4.65),(11.4,4.1))
    ax.plot([11.4,11.4,2.1],[1.75,1.25,1.25],color='#456e80',lw=1.5)
    arrow((2.1,1.25),(2.1,1.75))
    ax.text(6.5,1.04,'New measurements refine scene variables, not pretrained law weights',ha='center',fontsize=10,color='#456e80')
    ax.text(.35,.55,'Current scope: quasistatic solids; known geometry / categories; compliant attachments. Normal-contact extension uses an explicit penalty law.',fontsize=10,color='#4c616c')
    ax.text(.35,.24,'This is not an image encoder or an unconstrained next-frame network. Reactions are predictions; the main six-object study observes positions only.',fontsize=10,color='#4c616c')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    for suffix in ('png','svg'):fig.savefig(OUT/f'architecture-overview.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
