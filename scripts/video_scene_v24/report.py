import json
import numpy as np
from .protocol import ROOT,BASE,REPORT,write

def main():
    d=json.load(open(REPORT/'results.json'));s=d['summary'];improvements={}
    for group in s:
        v=s[group]['visual'];b=s[group]['image_blind']
        improvements[group]={k:100*(1-v[k]/b[k])for k in v}
    write(REPORT/'interpretation.json',dict(improvements_over_image_blind_percent=improvements,
        conclusion='The visual model improves mean 3D position and held-camera projection over the matched image-blind control; OOD displacement error worsens, and privileged true-initial persistence is stronger on held objects. No overall physical or rendering superiority.',
        trained=True,learned_physics=False,estimated_cameras=False,calibrated_uncertainty=False,generated_rgb=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');groups=['train','validation','test'];names=['Known objects,\nuntrained suffix times','Validation objects','Held rope/package']
    methods=['visual','image_blind','privileged_initial_persistence'];colors=['#258a83','#df8a3a','#999999']
    for ax,metric,title in zip(axes,['tracking_mm','held_camera_reprojection_px'],['3D reconstructed-track error (mm) ↓','Camera 2 projected-track error (pixels) ↓']):
        for i,m in enumerate(methods):
            values=[s[g][m][metric]for g in groups];bars=ax.bar(np.arange(3)+(i-1)*.25,values,.25,color=colors[i],label=m.replace('_',' '));ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
        ax.set_xticks(np.arange(3),names);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.suptitle('First learned real-video scene-query prototype — reconstruction, not forecasting',fontweight='bold')
    fig.savefig(REPORT/'comparison.png',dpi=160);fig.savefig(REPORT/'comparison.svg');plt.close(fig)
    p=REPORT/'comparison.svg';p.write_text('\n'.join(l.rstrip()for l in p.read_text().splitlines())+'\n')
    print(json.dumps(improvements,indent=2))

if __name__=='__main__':main()
