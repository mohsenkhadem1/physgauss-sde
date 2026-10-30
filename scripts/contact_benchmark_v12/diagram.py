"""Publication schematic of the implemented guarded prediction path."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scripts.contact_benchmark_v12.common import REPORT

def main():
    fig,ax=plt.subplots(figsize=(8,5.1));ax.set_xlim(0,8);ax.set_ylim(0,5.1);ax.axis('off')
    def box(x,y,text,color='#e8eef4'):
        ax.add_patch(FancyBboxPatch((x,y),2.2,1.1,boxstyle='round,pad=.05,rounding_size=.08',facecolor=color,edgecolor='#526578',linewidth=1.))
        ax.text(x+1.1,y+.55,text,ha='center',va='center',fontsize=9,color='#172a3a',linespacing=1.35)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#526578',lw=1.4))
    ax.text(.15,4.73,'Position-driven prediction with a local volume safeguard',fontsize=11.5,weight='bold',color='#172a3a')
    positions=[(.2,3.05),(2.9,3.05),(5.6,3.05),(5.6,1.5),(2.9,1.5),(.2,1.5)]
    labels=['1. Initial shape +\nobserved position\nprefix','2. Independent prefix\nfit and validation;\nfull-prefix refit','3. Inherited FEM\nforecast of\nmaterial shape','4. Pose correction +\nbounded local\nshape residual','5. Volume guard\non the original\nFEM elements','6. Static contact* +\npredicted future\ngeometry']
    for i,((x,y),label) in enumerate(zip(positions,labels)):box(x,y,label,'#d7efe7' if i==4 else '#e8eef4')
    arrow((2.47,3.6),(2.8,3.6));arrow((5.17,3.6),(5.5,3.6));arrow((6.7,2.98),(6.7,2.68))
    arrow((5.53,2.05),(5.27,2.05));arrow((2.83,2.05),(2.47,2.05))
    ax.text(.15,.99,'Additional inputs: prescribed tool positions, initial scene geometry and gravity.',fontsize=8.5,color='#172a3a')
    ax.text(.15,.59,'* SoftVT contact evolves rigid pose; shape is frozen at separation.\nPhysTwin uses the guarded object-frame forecast.',fontsize=8,color='#415568',linespacing=1.4)
    ax.text(.15,.12,'All fitting uses the observed prefix. Future object positions are reserved for evaluation.',fontsize=8,color='#415568')
    fig.tight_layout(pad=.5)
    for extension in ['png','svg']:fig.savefig(REPORT/f'guarded-algorithm.{extension}',dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
