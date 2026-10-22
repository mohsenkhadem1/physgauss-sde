"""Local scientific summary; no raw arrays or rendered media are published."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def render(screen,output):
    d=json.loads(Path(screen).read_text())
    if not d['completed'] or len(d['captures'])!=2:raise ValueError('Completed two-case frozen screen required')
    fig,ax=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    ax[0].add_patch(Rectangle((2,1),6,1,facecolor='#e4f2e8',edgecolor='#1d7348',label='Original excitation target'))
    for r,color in zip(d['captures'],['#1463a3','#b64b29']):
        x=100*r['maximum_absolute_principal_stretch_strain'];y=1e3*r['peak_nonrigid_coordinate_RMS_m']
        ax[0].scatter(x,y,color=color,s=55,zorder=3)
        ax[0].annotate(f"{r['duration_s']:.1f} s: {x:.2f}%, {y:.3f} mm",(x,y),xytext=(0,14 if r['duration_s']==.4 else -24),textcoords='offset points',ha='center',fontsize=9)
    ax[0].set(xlim=(0,10),ylim=(0,2),xlabel='Peak absolute principal stretch strain (%)',ylabel='Peak nonrigid coordinate RMS (mm)',title='Real strain; bulk-motion gate still fails')
    ax[0].legend(loc='upper left',fontsize=8)
    ax[0].grid(alpha=.15)
    durations=[f"{r['duration_s']:.1f} s" for r in d['captures']]
    torques=[1e3*r['maximum_actual_moment_Nm'] for r in d['captures']]
    bars=ax[1].bar(durations,torques,color=['#1463a3','#b64b29'],width=.5)
    ax[1].bar_label(bars,fmt='%.3f',padding=4)
    ax[1].set(ylim=(0,.5),xlabel='New rest-central tension pulse',ylabel='Peak actual dynamic torque (mNm)',title='Rest alignment does not imply zero dynamic torque')
    ax[1].text(.5,.92,'Rest torque ≈ 0 for both cases',ha='center',transform=ax[1].transAxes,fontsize=9)
    fig.suptitle('Consistent surface loads — 128-tet development screen\nNeither case admitted to fine meshes; no forward-accuracy qualification',fontsize=12)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',required=True);p.add_argument('--output',required=True);a=p.parse_args();render(a.screen,a.output)
