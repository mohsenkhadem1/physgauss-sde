"""Standalone summary figure; exact values remain in comparison.json."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v07_study import OUT
s=json.loads((OUT/'comparison.json').read_text());fig,ax=plt.subplots(1,3,figsize=(15,4.6));colors=['#386cb0','#7f7f7f','#f39c34','#22865a'];labels=['Fixed pokes','Random pokes','Adaptive pokes','Stop/view/poke']
for (p,r),color,label in zip(s['policies'].items(),colors,labels):
 ax[0].plot(range(3),r['rmse_mm'],'o-',label=label,color=color)
 ax[1].plot(range(3),np.array(r['object_coverage'])*100,'o-',label=label,color=color)
ax[0].set(xlabel='Acquisitions allowed (stopping permitted)',ylabel='Held-load coordinate RMSE (mm)',title='24 fresh test objects',xticks=[0,1,2]);ax[0].legend(fontsize=8)
ax[1].axhline(90,color='black',ls='--',lw=1);ax[1].set(xlabel='Acquisitions allowed',ylabel='Whole-object coverage (%)',title='All points and noise repeats',xticks=[0,1,2],ylim=(0,104))
policies=['repeat_initial','fixed','random','adaptive_pokes'];means=[np.mean([r['rmse_mm'] for r in s['phystwin'] if r['policy']==p]) for p in policies];ax[2].bar(range(4),means,color=['#888888',*colors[:3]])
for r in s['phystwin']:ax[2].plot(policies.index(r['policy']),r['rmse_mm'],'k.',alpha=.65)
ax[2].set(xticks=range(4),xticklabels=['Repeat\ninitial','Fixed','Random','Adaptive'],ylabel='Held-load coordinate RMSE (mm)',title='PhysTwin: 6 objects, matched updates')
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.2)
r=s['policies']['cost_aware'];fig.text(.5,.015,f"Cost-aware mean cost: {r['mean_cost']:.2f} versus 2.00 for two pokes. Assumed costs: view 0.25, poke 1.00. Static reference holds; synthetic RGB-D markers.",ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'comparison.png',dpi=160)
