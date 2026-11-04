"""Publication layout pass; geometry is unchanged, surfaces use initial convex-hull triangles."""
import generate_figures as g
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
# Fixed initial surface connectivity, transported by the saved node predictions.
tri=ConvexHull(g.seq[0][0]).simplices
fig=plt.figure(figsize=(7,2.05));allp=np.concatenate([a[g.t]for a in g.seq]);center=(allp.min(0)+allp.max(0))/2;radius=np.ptp(allp,axis=0).max()*.55
for i,(a,name,col)in enumerate(zip(g.seq,['Simulator reference','Without slip correction','Ours: slip-aware prediction'],[g.GRAY,g.ORANGE,g.TEAL])):
 ax=fig.add_subplot(1,3,i+1,projection='3d');pts=a[g.t];polys=pts[tri];normal=np.cross(polys[:,1]-polys[:,0],polys[:,2]-polys[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
 import matplotlib.colors as mc
 shade=.60+.4*np.abs(normal@np.array([.3,-.4,.866]));rgb=np.array(mc.to_rgb(col));colors=np.clip(rgb[None]*shade[:,None],0,1)
 if i:ax.add_collection3d(Poly3DCollection(g.seq[0][g.t][tri],facecolors='#dbe2e9',edgecolors='#a9b5c3',linewidths=.25,alpha=.16))
 ax.add_collection3d(Poly3DCollection(polys,facecolors=colors,edgecolors=(1,1,1,.34),linewidths=.25,alpha=.93))
 for k in range(3):getattr(ax,'set_'+['x','y','z'][k]+'lim')(center[k]-radius,center[k]+radius)
 ax.view_init(24,-60);ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(name,fontsize=9,pad=-4,color=col,fontweight='bold')
 if i:ax.text2D(.5,.025,f'Frame coordinate RMSE: {np.sqrt(np.mean((pts-g.seq[0][g.t])**2))*1000:.1f} mm',ha='center',transform=ax.transAxes,fontsize=7.5)
fig.subplots_adjust(left=0,right=1,top=.86,bottom=.025,wspace=-.05);g.save(fig,'teaser')
# Visual diagram: material-centered field, tool-relative attachments, world support.
fig=plt.figure(figsize=(7.1,3.05));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
cols=[(.015,.285,'Material coordinates',g.TEAL),(.355,.285,'Tool-relative attachments',g.ORANGE),(.695,.285,'World-space interactions',g.BLUE)]
for x,w,title,col in cols:
 ax.add_patch(FancyBboxPatch((x,.49),w,.47,boxstyle='round,pad=.006,rounding_size=.014',fc='#f0f4f8',ec='none'))
 ax.text(x+w/2,.925,title,ha='center',va='top',fontsize=8.5,fontweight='bold',color=col)
 for theta in np.linspace(0,2*np.pi,14):
  for phi in np.linspace(-1.0,1.0,7):
   u=np.cos(theta)*np.cos(phi);v=np.sin(theta)*np.cos(phi);z=np.sin(phi)
   px=x+w/2+.072*u+.018*v;py=.735+.073*z+.015*v
   ax.scatter([px],[py],s=3+2*(v+1),c=[col],alpha=.3+.6*(v+1)/2,edgecolors='none')
 if x<.1:
  ax.text(x+w/2,.59,'Bulk energy W(F, z)',ha='center',fontsize=8)
  ax.text(x+w/2,.535,'Properties move with the object',ha='center',fontsize=6.8)
 elif x<.5:
  for sign in [-1,1]:
   xx=x+w/2+sign*.10;ax.plot([xx,xx],[.68,.79],color=g.INK,lw=4);ax.plot([xx,x+w/2+sign*.062],[.735,.735],color=col,lw=1.4,ls=':')
  ax.text(x+w/2,.59,'Attachment energy + slip state',ha='center',fontsize=8)
  ax.text(x+w/2,.535,'Known tool motion; fitted coupling',ha='center',fontsize=6.8)
 else:
  ax.plot([x+.065,x+w-.065],[.65,.65],c=g.BLUE,lw=2);ax.add_patch(FancyArrowPatch((x+w-.055,.81),(x+w-.055,.69),arrowstyle='-|>',mutation_scale=10,color=g.BLUE))
  ax.text(x+w/2,.59,'Gravity + environmental contact',ha='center',fontsize=8)
  ax.text(x+w/2,.535,'Depend on current world position',ha='center',fontsize=6.8)
ax.text(.5,.435,'Observed prefix → fit scene parameters → freeze → query future tool positions',ha='center',fontsize=8.3,fontweight='bold')
for x,w,title,body in [(.02,.44,'SoftVTBench','Analytic shape + position correction\nSlip / release + reduced contact rollout'),(.54,.44,'PhysTwin','Frozen neural bulk + scene descriptors\nEquilibrium + shared Gaussian appearance')]:
 ax.add_patch(FancyBboxPatch((x,.065),w,.265,boxstyle='round,pad=.009,rounding_size=.014',fc='#e6f2f0',ec='none'));ax.text(x+w/2,.28,title,ha='center',fontweight='bold',fontsize=9);ax.text(x+w/2,.205,body,ha='center',va='top',fontsize=8,linespacing=1.5)
 ax.add_patch(FancyArrowPatch((x+w/2,.405),(x+w/2,.34),arrowstyle='-|>',mutation_scale=9,color=g.INK))
ax.text(.5,.012,'Common physical roles; benchmark-specific adapters. Diagrams are schematic.',ha='center',fontsize=7,color=g.GRAY);g.save(fig,'method')
# Main-paper single-column plots need readable type at final size.
fig,ax=plt.subplots(figsize=(3.4,2.35));r=g.soft['contact_scene_state'];x=np.array([a['baseline_mm']for a in r['per_case']]);y=np.array([a['proposal_mm']for a in r['per_case']]);lim=max(x.max(),y.max())*1.08
ax.plot([0,lim],[0,lim],c=g.GRAY,lw=1,ls='--');ax.scatter(x,y,c=np.where(y<x,g.TEAL,g.ORANGE),s=20,zorder=3);ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Without slip correction (mm)',ylabel='Ours (mm)');ax.text(.04,.94,'36 improve / 4 worsen',transform=ax.transAxes,va='top',fontsize=8);ax.grid(alpha=.12);fig.tight_layout(pad=.4);g.save(fig,'soft_scatter')
fig,ax=plt.subplots(figsize=(3.4,2.15));s=g.transfer['summary']['initial_geometry_gate_passed'];xx=np.arange(3);w=.24
for i,(m,name,col)in enumerate([('neural','Ours',g.TEAL),('analytic','Analytic',g.BLUE),('native_frozen','Frozen PhysTwin',g.ORANGE)]):
 vals=[s[m][k]for k in ['tracking_mm','displacement_mm','pair_distance_change_mm']];b=ax.bar(xx+(i-1)*w,vals,w,color=col,label=name);ax.bar_label(b,fmt='%.1f',padding=2,fontsize=6.5)
ax.set(xticks=xx,xticklabels=['Position','Displacement','Pair change'],ylabel='Error (mm)',ylim=(0,57));ax.legend(fontsize=6.8,ncol=3,loc='upper center',frameon=False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);fig.tight_layout(pad=.4);g.save(fig,'transfer_compact')
import json
p=g.HERE/'evidence/figure-sources.json';d=json.loads(p.read_text());d['selection']['soft_teaser']['surface']='Initial node convex-hull connectivity used only for visualization; not simulator element topology. All vertex positions are unaltered.';p.write_text(json.dumps(d,indent=2)+'\n')
