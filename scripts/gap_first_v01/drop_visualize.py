"""Actual native render contact sheet and comparison video; no geometry synthesis."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for font in Path("/home/bayespc2/.local/share/fonts/arial").glob("*.TTF"): font_manager.fontManager.addfont(str(font))
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));from scripts.gap_first_v01.drop_protocol import verify
BASE=ROOT/'runs/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop';OUT=ROOT/'output/gap-first-v01/drop'
plt.rcParams.update({'font.family':'Arial','font.size':12,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42})
WIDE='--wide' in sys.argv
STEM='native-drop-wide-comparison' if WIDE else 'native-drop-comparison'
FONT='/home/bayespc2/.local/share/fonts/arial/Arial.TTF'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(asset,case):
 return np.load(BASE/'cases'/asset/case/('illustration-camera-v2.npz' if WIDE else 'trajectory.npz'))['rgb'] if WIDE else np.load(BASE/'cases'/asset/case/'trajectory.npz')['rgb'][:,0]
def main():
 p=verify();assets=p['assets'];cases=['passive-A','passive-B','changed-A','changed-B'];names=['Pastry','Shell','Bottle'];images={};records=[];traces={}
 for asset in assets:
  for case in cases:
   path=BASE/'cases'/asset/case/'result.json';d=json.loads(path.read_text());assert d['complete'];images[(asset,case)]=load(asset,case)
   cam=np.load(BASE/'cases'/asset/case/'illustration-camera-v2.npz') if WIDE else np.load(BASE/'initial/cameras.npz');extr=cam['extrinsics'] if WIDE else cam['extrinsics'][0];intr=cam['intrinsics'] if WIDE else cam['intrinsics'][0];xyz=np.load(BASE/'cases'/asset/case/'trajectory.npz')['centroids']-.5;camera_xyz=xyz@extr[:3,:3].T+extr[:3,3];projected=camera_xyz@intr.T;traces[(asset,case)]=projected[:,:2]/projected[:,2,None]*512
   records.append(dict(asset=asset,case=case,result_sha256=sha(path),trajectory_sha256=sha(BASE/'cases'/asset/case/'trajectory.npz')))
 fig,axes=plt.subplots(3,4,figsize=(12,8.3),gridspec_kw={'wspace':.035,'hspace':.12});fig.subplots_adjust(left=.035,right=.995,bottom=.070,top=.90)
 headings=['Gravity only: A','Gravity only: B','Added force: A','Added force: B']
 for row,asset in enumerate(assets):
  analysis=json.loads((REPORT/f'{asset}-analysis.json').read_text())
  for col,case in enumerate(cases):
   ax=axes[row,col];frame=18 if col<2 else 24;ax.imshow(images[(asset,case)][frame]);ax.set_xticks([]);ax.set_yticks([])
   for spine in ax.spines.values():spine.set_visible(False)
   if row==0:ax.set_title(headings[col],fontweight='bold',fontsize=13,pad=8)
   if col==0:ax.set_ylabel(names[row],fontsize=13,fontweight='bold',labelpad=7)
   if col==1:ax.set_xlabel('Same passive observations, t = 0.54 s',fontsize=10,labelpad=3)
   if col==3:ax.set_xlabel(f"Trajectory RMS gap: {analysis['changed_force_target_pair_rms_difference']:.3f}",fontsize=10,labelpad=3)
 fig.text(.5,.016,'Passive t=0.54 s; force t=0.72 s. Normalized scene units; uniform body force 20 per anchor (2048 anchors).',ha='center',fontsize=10)
 fig.suptitle('Same drop, different physical response',fontsize=19,fontweight='bold',y=.985)
 fig.text(.5,.94,'A: mass 1, stiffness 300, damping 1     |     B: mass 2, stiffness 600, damping 2',ha='center',fontsize=12)
 OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/(STEM+'.png'),dpi=220,facecolor='white');fig.savefig(OUT/(STEM+'.pdf'),facecolor='white');plt.close(fig)
 font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,19);big=ImageFont.truetype(FONT,29);tile=352;width=4*tile;height=3*(tile+28)+115
 writer=imageio.get_writer(OUT/(STEM+'.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
 for frame in range(30):
  canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);draw.text((16,8),'Same drop movie, different response to the same added force',font=big,fill='black')
  draw.text((16,45),f'Simulation time {frame*.03:.2f} s   |   Playback slowed 3.3x   |   Force active: '+('YES' if 12<=frame<=17 else 'no'),font=small,fill='black')
  for col,h in enumerate(headings):draw.text((col*tile+12,78),h,font=font,fill='black')
  for row,asset in enumerate(assets):
   y=112+row*(tile+28)
   for col,case in enumerate(cases):
    canvas.paste(Image.fromarray(images[(asset,case)][frame]).resize((tile,tile),Image.Resampling.LANCZOS),(col*tile,y))
    uv=traces[(asset,case)][max(0,frame-10):frame+1]*tile/512;xy=[(float(col*tile+q[0]),float(y+q[1])) for q in uv if 0<=q[0]<tile and 0<=q[1]<tile]
    if len(xy)>1:draw.line(xy,fill=(30,175,175),width=2)
    if xy:
     xx,yy=xy[-1];draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill='white',outline=(30,175,175),width=2)
    if col>=2 and 12<=frame<=17:
     xx=col*tile+tile-28;yy=y+tile//2;draw.line([(xx,yy+35),(xx,yy-35)],fill=(28,155,155),width=5);draw.polygon([(xx,yy-45),(xx-10,yy-28),(xx+10,yy-28)],fill=(28,155,155))
   draw.text((12,y+tile+1),names[row]+'   |   Native RGB; shared camera/scale; line = model centroid trace',font=small,fill='black')
  writer.append_data(np.asarray(canvas))
 writer.close()
 manifest=dict(protocol_sha256=sha(REPORT/'protocol.json'),generator_sha256=sha(__file__),inputs=records,snapshots=dict(passive=18,changed_force=24),camera=('sharedillustrationcamera' if WIDE else 0),render_source='Actual native GaussianRenderer RGB; no generated geometry or deformation exaggeration.',movie=dict(fps=10,physical_dt=.03,playback_slowdown=10/3,response_overlay='Projected last11modelcontrol-centroids, not an observed surface marker.',force_overlay='Upward arrow only duringknown+z20peranchor window.'),labels='Mass/force per control anchor; normalized scene units. Response gap is21coordinate centroidRMS overframes18..24, not singleframe displacement.',artifacts={f.name:sha(f) for f in OUT.glob(STEM+'.*')})
 (OUT/(STEM+'-manifest.json')).write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
