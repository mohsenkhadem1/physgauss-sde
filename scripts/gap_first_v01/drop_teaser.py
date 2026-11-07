"""First-predeclared-asset teaser: common crop, unchanged native RGB."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for font in Path("/home/bayespc2/.local/share/fonts/arial").glob("*.TTF"): font_manager.fontManager.addfont(str(font))
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-first-v01/drop';OUT=ROOT/'output/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop'
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':'black','pdf.fonttype':42})
p=json.loads((REPORT/'protocol.json').read_text());a=p['assets'][0];cases=['passive-A','passive-B','changed-A','changed-B'];paths=[BASE/'cases'/a/c/'illustration-camera-v2.npz' for c in cases];rgb=[np.load(f)['rgb'] for f in paths]
mask=np.any(np.stack([np.max(x,axis=-1)>8 for x in rgb]),axis=(0,1));yy,xx=np.where(mask);x0=max(0,xx.min()-15);x1=min(512,xx.max()+16);y0=max(0,yy.min()-15);y1=min(512,yy.max()+16)
fig,axs=plt.subplots(1,4,figsize=(7.6,3.3));fig.subplots_adjust(left=.005,right=.995,bottom=.16,top=.83,wspace=.035)
for i,ax in enumerate(axs):
 ax.imshow(rgb[i][18 if i<2 else 24,y0:y1,x0:x1]);ax.axis('off');ax.set_title(['Passive A','Passive B','Body force A','Body force B'][i],fontweight='bold')
fig.text(.5,.95,'Same observed drop; different response to the same force',ha='center',fontsize=12,fontweight='bold')
fig.text(.5,.06,'A: (m,k,d)=(1,300,1); B: (2,600,2). Passive: 0.54 s; forced: 0.72 s.',ha='center',fontsize=8.5)
fig.text(.5,.008,'Native scene units; force +20 per anchor. One camera and one common crop; no motion exaggeration.',ha='center',fontsize=8.5)
for ext in ['png','pdf']:fig.savefig(OUT/('native-drop-teaser.'+ext),dpi=220,facecolor='white')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
(OUT/'native-drop-teaser-manifest.json').write_text(json.dumps(dict(asset=a,selection='First predeclared asset; no outcome selection',crop_xyxy=list(map(int,[x0,y0,x1,y1])),crop_definition='Union foreground across all four conditions and all thirty frames, plus 15 pixels; same crop in every panel',source_sha256=sha(__file__),camera_sha256=sha(REPORT/'illustration-camera-v2.json'),inputs={str(f.relative_to(ROOT)):sha(f) for f in paths},outputs={ext:sha(OUT/('native-drop-teaser.'+ext)) for ext in ['png','pdf']}),indent=2))
