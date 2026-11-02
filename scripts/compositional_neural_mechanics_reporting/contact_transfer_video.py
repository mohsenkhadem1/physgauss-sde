"""Reporting-only camera orbit of final saved equilibria; no motion synthesis."""
import json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio_ffmpeg
from scripts.compositional_neural_mechanics.contact_transfer import ROOT,R,OUT,PLAN,FAMILIES,verify,sha,write

def main():
 p=json.loads(PLAN.read_text());verify(p);rp=R/'contact-transfer-fine.json';report=json.loads(rp.read_text())
 if not report.get('terminal')or not report['complete']or report['plan_sha256']!=sha(PLAN):raise ValueError('Complete provenance-bound result required')
 folder=R/'videos';folder.mkdir(exist_ok=True);frames=folder/'contact-transfer-orbit-frames';frames.mkdir(exist_ok=True);data={};hashes={str(rp.relative_to(ROOT)):sha(rp)}
 for family in FAMILIES:
  for method in ['analytic_teacher','shared_laws']:
   key=f'{family}-{method}-n8-q32';row=next(x for x in report['rows']if x['key']==key);f=OUT/'states'/(key+'.npz')
   if not row['completed']or sha(f)!=row['response_sha256']:raise ValueError('Invalid rendered state')
   data[family,method]=np.load(f)['surface_frames'][-1];hashes[str(f.relative_to(ROOT))]=sha(f)
 fig=plt.figure(figsize=(10,10),dpi=120);axes=[];theta=np.linspace(0,2*np.pi,32);phi=np.linspace(0,np.pi,20);sphere=np.stack([.12+.045*np.outer(np.cos(theta),np.sin(phi)),.045*np.outer(np.sin(theta),np.sin(phi)),.080+.045*np.outer(np.ones_like(theta),np.cos(phi))],-1)
 for i,family in enumerate(FAMILIES):
  error=report['transfer_comparisons'][family]['field_rms_m']*1e6
  for j,method in enumerate(['analytic_teacher','shared_laws']):
   ax=fig.add_subplot(2,2,2*i+j+1,projection='3d');axes.append(ax)
   for xyz in data[family,method]:ax.plot_surface(*np.moveaxis(xyz,-1,0),color='#4096bd'if j==0 else'#65ad78',linewidth=0,antialiased=False)
   ax.plot_surface(*np.moveaxis(sphere,-1,0),color='#d5944b',alpha=.35,linewidth=0)
   xx,yy=np.meshgrid([-.02,.26],[-.06,.06]);ax.plot_surface(xx,yy,np.full_like(xx,-.04001),color='gray',alpha=.25)
   # Fixed bilateral target planes are guides, not additional collision bodies.
   for x in [0,.24]:ax.plot([x]*5,[-.04,.04,.04,-.04,-.04],[-.04,-.04,.04,.04,-.04],color='#333333',linestyle='--',linewidth=1.1)
   ax.set(xlim=(-.02,.26),ylim=(-.06,.06),zlim=(-.055,.13));ax.set_box_aspect((.28,.12,.185));ax.set_xticks([0,.12,.24]);ax.set_yticks([-.04,.04]);ax.set_zticks([-.04,.04,.12]);ax.tick_params(labelsize=7);ax.set_xlabel('x (m)',fontsize=8);ax.set_ylabel('y (m)',fontsize=8);ax.set_zlabel('z (m)',fontsize=8)
   label='Analytic teacher (simulation)'if j==0 else'Frozen learned bulk + interface'
   ax.set_title(f'{family.replace("_"," + ")}\n{label}\nMatched field error: {error:.2f} µm',fontsize=9)
 fig.suptitle('Known sphere + floor contact with gravity\nSTATIC final equilibria: camera orbit only; physical scale',fontsize=13)
 fig.text(.5,.025,'Dashed end outlines: fixed bilateral attachment targets. No friction, impact, or real-object video.',ha='center',fontsize=9)
 fig.subplots_adjust(top=.83,bottom=.10,hspace=.6,wspace=.1)
 for index in range(48):
  for ax in axes:ax.view_init(elev=20+4*np.sin(2*np.pi*index/48),azim=-65+50*np.sin(2*np.pi*index/48))
  fig.savefig(frames/f'{index:03d}.png')
 plt.close(fig);video=folder/'contact-transfer-orbit.mp4';ff=imageio_ffmpeg.get_ffmpeg_exe();subprocess.run([ff,'-y','-hide_banner','-loglevel','error','-framerate','12','-i',str(frames/'%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
 subprocess.run([ff,'-hide_banner','-loglevel','error','-i',str(video),'-f','null','-'],check=True)
 reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);count=sum(1 for _ in reader)
 if count!=48 or abs(meta['duration']-4)>0.05:raise ValueError('Video frame/duration mismatch')
 write(folder/'contact-transfer-orbit.json',dict(source_sha256=sha(Path(__file__)),plan_sha256=sha(PLAN),input_sha256=hashes,video_sha256=sha(video),video_metadata=meta,decoded_frames=count,decode_passed=True,scope='Both declared families; final n8q32 static equilibria. Camera motion only, no interpolation of physical motion, no new solves.',preview=str((frames/'000.png').relative_to(ROOT))))
if __name__=='__main__':main()
