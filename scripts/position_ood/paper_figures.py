"""Standalone scientific figures for the manuscript (no generated measurements)."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.position_ood.runtime import BASE
OUT=ROOT/'papers'/BASE.name/'figures';OUT.mkdir(exist_ok=True)

def overview():
    fig,ax=plt.subplots(figsize=(7.1,2.65));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=color,edgecolor='#455A64',linewidth=.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#455A64',linewidth=.9))
    box(.1,1.05,2.,1.1,'Observed prefix\nRGB-D positions\nTool poses','#E6F0F7');box(2.8,1.65,2.2,.65,'Material field\nRest coordinates X','#E7F1E8');box(2.8,.55,2.2,.75,'Environmental field\nSpatial coordinates x','#FFF0D8');box(5.65,1.,1.7,1.15,'Nonlinear\nequilibrium\nImplicit gradients','#EDE9F5');box(8.,1.1,1.8,.95,'New tool poses\nPredicted shape\nCalibrated intervals','#E6F0F7')
    arrow((2.2,1.6),(2.7,2.));arrow((2.2,1.5),(2.7,.95));arrow((5.1,2),(5.55,1.8));arrow((5.1,.95),(5.55,1.35));arrow((7.45,1.6),(7.9,1.6));ax.plot([8.9,8.9,1.1],[.95,.3,.3],color='#52606D',linewidth=.9);arrow((1.1,.3),(1.1,.95));ax.text(5.,.06,'Select one positional probe; update from its observed response',ha='center',fontsize=8,color='#334E68')
    fig.savefig(OUT/'method.pdf',bbox_inches='tight');fig.savefig(OUT/'method.png',dpi=220,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':overview()
