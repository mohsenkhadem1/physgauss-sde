import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();base=root/'runs/physical-refinement-v15/development';report=root/'reports/physical-refinement-v15'
world='development-3';action='design_opposed_translation';camera='left';paths=[base/'world-render-bank'/world/f'n{n}'/action/f'{camera}.npz'for n in (4,6)]
images=[]
for p in paths:
 with np.load(p)as a:images.append({k:a[k].copy()for k in ('rgb','depth')})
jpath=report/'supersampling-diagnostic.json';j=json.loads(jpath.read_text());row=next(r for r in j['rows']if (r['world'],r['action'],r['camera'])==(world,action,camera))
a,b=images;ma=np.isfinite(a['depth']);mb=np.isfinite(b['depth']);changed=ma!=mb
fig,ax=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
for col,(im,n)in enumerate(zip(images,(4,6))):
 ax[0,col].imshow(im['rgb']);ax[0,col].set_title(f'Actual rendered RGB — n{n}')
 d=np.where(np.isfinite(im['depth']),im['depth'],np.nan)
 m=ax[1,col].imshow(d,cmap='viridis',vmin=.28,vmax=.57);ax[1,col].set_title(f'Actual optical depth — n{n} (m)')
 for r in (0,1):ax[r,col].axis('off')
fig.colorbar(m,ax=ax[1,:2],shrink=.65,label='Camera-axis depth (m); white = no return')
ax[0,2].imshow(np.clip(np.abs(a['rgb']-b['rgb'])*100,0,1));ax[0,2].set_title('RGB absolute difference ×100\nDisplay amplification only');ax[0,2].axis('off')
overlay=a['rgb']*.35
# Enlarge each changed-pixel marker for visibility, without changing data/counts.
ax[1,2].imshow(overlay);ys,xs=np.where(changed);ax[1,2].scatter(xs,ys,s=70,facecolors='none',edgecolors='#ff00ff',linewidths=1.5)
ax[1,2].set_title(f'Silhouette difference: {changed.sum()} pixels\nMagenta rings enlarge pixel locations');ax[1,2].axis('off')
factors=[1,2,4];mesh=[row['mesh_D2'][str(s)]for s in factors]
ax[0,3].plot(factors,mesh,'o-',color='#245ca6');ax[0,3].set_yscale('log');ax[0,3].set_xticks(factors);ax[0,3].set_xlabel('Supersampling factor');ax[0,3].set_ylabel('Mahalanobis squared distance D²');ax[0,3].set_title('Same-world n4 vs n6 feature mismatch');ax[0,3].grid(alpha=.2)
for x,y in zip(factors,mesh):ax[0,3].annotate(f'{y:.4g}',(x,y),xytext=((-66 if x==4 else 4),6),textcoords='offset points',fontsize=9)
vals=[row['fixed_geometry_renderer_D2'][f'n{n}_{pair}']for n in (4,6)for pair in ('1_to_2','2_to_4')]
ax[1,3].bar(['n4: 1→2','n4: 2→4','n6: 1→2','n6: 2→4'],vals,color=['#a26740','#cc965e']*2);ax[1,3].tick_params(axis='x',rotation=25);ax[1,3].set_ylabel('D², unchanged covariance');ax[1,3].set_title('Fixed geometry: renderer still unconverged')
fig.suptitle('Development numerical observation mismatch — opposed translation, left view\nSame physical parameters; original images unaltered. Improved mesh agreement does not establish renderer convergence.',fontsize=14)
outputs=[]
for ext in ('png','svg'):
 p=report/'figures'/f'renderer-numerical-discrepancy.{ext}';fig.savefig(p,dpi=160);outputs.append(p)
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=dict(scope='Development-only diagnostic, no renderer/bank modifications. Original factor1 RGB/depth shown; only difference displays amplified.',case=dict(world=world,action=action,camera=camera),input_sha256={str(p.relative_to(root)):sha(p)for p in paths+[jpath]},output_sha256={str(p.relative_to(root)):sha(p)for p in outputs},mesh_D2=row['mesh_D2'],fixed_geometry_renderer_D2=row['fixed_geometry_renderer_D2'],silhouette_changed_pixels=int(changed.sum()))
(report/'figures/renderer-numerical-discrepancy-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
